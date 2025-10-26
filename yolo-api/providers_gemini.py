# providers_gemini.py
import os
import json
import time
from typing import Any, Dict, Callable, List
import google.generativeai as genai

# ================== Config ==================
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Falta GOOGLE_API_KEY en el entorno")

genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = (
    "Eres un analista de tránsito. Nunca inventes números: usa las herramientas "
    "para obtener datos reales de la base. "
    "SIEMPRE incluye 'video_id' en cualquier llamada de función. "
    "Responde claro y conciso con los resultados de las herramientas."
)

# =============== Tools Declaration ===========
def _tool_decl():
    return {
        "function_declarations": [
            {
                "name": "get_summary",
                "description": "Resumen general del video.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "video_id": {"type": "STRING"},
                    },
                    "required": ["video_id"],
                },
            },
            {
                "name": "get_colors",
                "description": "Distribución de colores por clase.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "video_id": {"type": "STRING"},
                    },
                    "required": ["video_id"],
                },
            },
            {
                "name": "get_tracks",
                "description": "Tracks y su duración (frames), filtrable por clase.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "video_id": {"type": "STRING"},
                        "klass": {"type": "STRING"},
                    },
                    "required": ["video_id"],
                },
            },
            {
                "name": "get_trajectory",
                "description": "Trayectoria (cx,cy) del track dado.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "video_id": {"type": "STRING"},
                        "track_id": {"type": "INTEGER"},
                    },
                    "required": ["video_id", "track_id"],
                },
            },
            {
                "name": "get_speed",
                "description": "Velocidad media y pico (px/s) del track dado.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "video_id": {"type": "STRING"},
                        "track_id": {"type": "INTEGER"},
                    },
                    "required": ["video_id", "track_id"],
                },
            },
        ]
    }

# =============== Helpers =====================
def _extract_function_calls_from_parts(parts: List[Any]) -> List[Dict[str, Any]]:
    """
    Normaliza las function calls de Gemini en una lista:
    [{"name": str, "args": dict}, ...]
    Soporta varias formas (function_call, args_json, args tipo lista de pares, etc.).
    """
    calls: List[Dict[str, Any]] = []
    for p in parts or []:
        fc = getattr(p, "function_call", None)
        if not fc:
            # A veces viene como dict directamente
            if isinstance(p, dict) and "functionCall" in p:
                fc = p["functionCall"]
            elif isinstance(p, dict) and "function_call" in p:
                fc = p["function_call"]
            else:
                continue

        # name
        name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else None)
        if not name:
            continue

        args: Dict[str, Any] = {}

        # args pueden venir en distintas formas:
        if hasattr(fc, "args") and fc.args is not None:
            # fc.args puede ser dict o lista de objetos con key/value
            if isinstance(fc.args, dict):
                args = dict(fc.args)
            else:
                # lista de pares
                try:
                    args = {item.key: item.value for item in fc.args}
                except Exception:
                    # fallback: str -> intentar json
                    try:
                        args = json.loads(fc.args)
                    except Exception:
                        args = {}
        elif hasattr(fc, "args_json") and fc.args_json:
            try:
                args = json.loads(fc.args_json)
            except Exception:
                args = {}
        elif isinstance(fc, dict):
            # dict-based
            raw_args = fc.get("args")
            if isinstance(raw_args, dict):
                args = dict(raw_args)
            elif isinstance(raw_args, list):
                # lista de {key, value}
                try:
                    args = {kv["key"]: kv["value"] for kv in raw_args if "key" in kv and "value" in kv}
                except Exception:
                    args = {}
            elif isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {}

        calls.append({"name": name, "args": args})

    return calls

# =============== Main Orchestrator ===========
# ... arriba igual ...

def _coerce_args(fc) -> Dict[str, Any]:
    """Convierte los args de la function_call del formato que venga a dict."""
    # Casos posibles con el SDK:
    # - fc.args ya es dict
    # - fc.args es una lista de pares key/value (p.ej. objetos con .key/.value)
    # - fc.args_json es un string JSON
    if hasattr(fc, "args") and isinstance(fc.args, dict):
        return dict(fc.args)
    if hasattr(fc, "args") and isinstance(fc.args, list):
        out = {}
        for item in fc.args:
            # soporta item como dict {'key':..., 'value':...} o como objeto con attrs
            if isinstance(item, dict) and "key" in item and "value" in item:
                out[item["key"]] = item["value"]
            else:
                k = getattr(item, "key", None)
                v = getattr(item, "value", None)
                if k is not None:
                    out[k] = v
        return out
    if hasattr(fc, "args_json") and fc.args_json:
        try:
            return json.loads(fc.args_json)
        except Exception:
            return {}
    return {}

def gemini_plan_and_answer(question: str, video_id: str, tool_executor):
    tools = _tool_decl()
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        tools=tools,
        system_instruction=SYSTEM_PROMPT,
    )

    start = time.time()
    user_msg = {"role": "user", "parts": [{"text": f"video_id={video_id}\nPregunta: {question}"}]}
    resp = model.generate_content(
        contents=[user_msg],
        safety_settings=None,
        generation_config={"temperature": 0.2},
    )

    tool_calls = []
    messages = [user_msg]

    if hasattr(resp, "candidates") and resp.candidates:
        cand = resp.candidates[0]
        parts = cand.content.parts if cand.content else []
        for p in parts:
            fc = getattr(p, "function_call", None) or getattr(p, "functionCall", None)
            if not fc:
                continue

            name = getattr(fc, "name", None)
            args = _coerce_args(fc)

            # inyecta video_id si la herramienta lo requiere y no vino
            if name in {"get_summary", "get_colors", "get_tracks", "get_trajectory", "get_speed"}:
                args.setdefault("video_id", video_id)

            result = tool_executor(name, args)
            tool_calls.append({"name": name, "args": args, "result_preview": str(result)[:300]})

            # *** CLAVE CORREGIDA: function_response (snake_case) ***
            messages.append({
                "role": "tool",
                "parts": [{
                    "function_response": {
                        "name": name,
                        "response": {"content": result}
                    }
                }]
            })

        if len(messages) > 1:
            final = model.generate_content(
                contents=messages,
                safety_settings=None,
                generation_config={"temperature": 0.2},
            )
            end = time.time()
            usage = getattr(final, "usage_metadata", None)
            answer = (final.text or "").strip()
            return {
                "answer": answer if answer else "[Sin respuesta]",
                "latency_ms": int((end - start) * 1000),
                "tool_calls": tool_calls,
                "tokens": {
                    "prompt": getattr(usage, "prompt_token_count", None),
                    "candidates": getattr(usage, "candidates_token_count", None),
                    "total": getattr(usage, "total_token_count", None),
                },
            }

    end = time.time()
    usage = getattr(resp, "usage_metadata", None)
    answer = (getattr(resp, "text", "") or "").strip()
    return {
        "answer": answer if answer else "[Sin respuesta]",
        "latency_ms": int((end - start) * 1000),
        "tool_calls": tool_calls,
        "tokens": {
            "prompt": getattr(usage, "prompt_token_count", None),
            "candidates": getattr(usage, "candidates_token_count", None),
            "total": getattr(usage, "total_token_count", None),
        },
    }
