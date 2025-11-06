# yolo-api/metrics/eval_runner_gemini.py
import os, json, math, time
from pathlib import Path
import pandas as pd
from jsonschema import validate, ValidationError
from sqlalchemy import text
from dotenv import load_dotenv
import google.generativeai as genai

from database import engine  # usa tu engine actual (Session Pooler)

# ====== Configuración base (usa lo mismo que tu providers_gemini.py) ======
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Falta GOOGLE_API_KEY en el entorno")
genai.configure(api_key=API_KEY)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # mismo default que ya usas
TEMPERATURE = 0.1

SCHEMA_PATH = Path(__file__).with_name("schema.json")
DATASET_PATH = Path(__file__).with_name("dataset.jsonl")
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

SYSTEM_ES = """Eres un analista de datos de tráfico.
Responde SOLO en JSON válido exacto al siguiente esquema:
{"answer": string, "type":"numeric|text|sql", "value": number|string|null,
 "sql_used": string|null, "evidence": string[], "confidence": number}
- Si hace falta consultar BD, escribe el SQL en 'sql_used'.
- Si no hay evidencia, responde EXACTAMENTE:
{"answer":"insufficient_evidence","type":"text","value":null,"sql_used":null,"evidence":[],"confidence":0.0}
"""

# ====== Helpers de BD / evaluación ======
def _run_sql(sql: str):
    """Ejecuta SQL y devuelve todas las filas (lista de tuplas)."""
    with engine.connect() as conn:
        res = conn.execute(text(sql))
        try:
            return res.fetchall()
        except Exception:
            return []

def _json_ok(obj):
    """Valida contra el schema JSON de respuesta del LLM."""
    try:
        validate(obj, SCHEMA)
        return True, ""
    except ValidationError as e:
        return False, str(e)

def _compare_numeric(pred, gold):
    """Devuelve (acc, mae) para métricas numéricas."""
    if isinstance(gold, (int, str)):
        # igualdad exacta si el gold es entero/cadena
        try:
            return int(float(pred) == float(gold)), abs(float(pred) - float(gold))
        except Exception:
            return 0, math.inf
    try:
        p = float(pred); g = float(gold)
        mae = abs(p - g)
        return (1 if mae < 1e-6 else 0), mae
    except Exception:
        return 0, math.inf

def _compare_sql(rows_pred, rows_gold):
    """1 si las filas coinciden exactamente (como string); 0 si no."""
    return int(str(rows_pred) == str(rows_gold))

def _grounded_ok(evidence_list):
    """1 si hay evidencia (lista no vacía), 0 si no."""
    return int(bool(evidence_list))

def _user_prompt(question: str, video_id: str):
    """Prompt de usuario que verá el modelo (estilo 'toolformer' pero simple)."""
    return f"""Pregunta: {question}
Contexto:
- video_id: {video_id}
- Tabla detections(video_id, frame_number, ts, object_class, confidence, x1,y1,x2,y2, track_id)
RESPONDE SOLO CON EL JSON DEL ESQUEMA INDICADO. Nada más.
"""

# ====== Llamada a Gemini (misma API key/modelo que tu app) ======
def _call_gemini(system: str, user: str, model_name: str = MODEL, temperature: float = TEMPERATURE):
    m = genai.GenerativeModel(model_name=model_name, system_instruction=system)
    t0 = time.time()
    resp = m.generate_content(
        [user],
        generation_config={"response_mime_type": "application/json", "temperature": temperature}
    )
    latency_ms = (time.time() - t0) * 1000.0

    # Texto de salida
    txt = getattr(resp, "text", None) or ""
    # usage_metadata es un objeto proto (NO dict)
    meta = getattr(resp, "usage_metadata", None)
    in_tok   = getattr(meta, "prompt_token_count", None) if meta else None
    out_tok  = getattr(meta, "candidates_token_count", None) if meta else None
    total_tok= getattr(meta, "total_token_count", None) if meta else None

    return txt, latency_ms, in_tok, out_tok


# ====== Runner principal ======
def run_eval(dataset_path: Path = DATASET_PATH):
    """Ejecuta el dataset contra Gemini y genera runs/per_question.csv y runs/summary.csv."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {dataset_path}")

    items = [json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows_out = []

    for it in items:
        lang = it.get("lang", "es")
        if lang != "es":
            # aquí podrías agregar SYSTEM_EN si luego quieres comparar idiomas
            pass

        system = SYSTEM_ES
        user   = _user_prompt(it["question"], it.get("video_id", ""))

        raw, latency_ms, in_tok, out_tok = _call_gemini(system, user, MODEL, TEMPERATURE)

        parsed = None; json_valid = 0; err = ""
        try:
            parsed = json.loads(raw)
            ok, err = _json_ok(parsed)
            json_valid = int(ok)
        except Exception as e:
            err = f"parse_error: {e}"

        numeric_acc = numeric_mae = sql_ok = grounded = halluc = None

        if json_valid and isinstance(parsed, dict):
            grounded = _grounded_ok(parsed.get("evidence", []))
            halluc   = 1 - grounded

            et = it["eval_type"]
            if et == "numeric":
                # Si el modelo propone SQL, intentamos ejecutarlo; si no, usamos value.
                if parsed.get("sql_used"):
                    try:
                        rows = _run_sql(parsed["sql_used"])
                        val = rows[0][0] if rows and len(rows[0]) == 1 else parsed.get("value")
                    except Exception:
                        val = parsed.get("value")
                else:
                    val = parsed.get("value")
                acc, mae = _compare_numeric(val, it["gold"])
                numeric_acc, numeric_mae = acc, mae

            elif et == "sql":
                pred_sql = parsed.get("sql_used")
                if pred_sql:
                    try:
                        rows_pred = _run_sql(pred_sql)
                        rows_gold = _run_sql(it["gold_sql"])
                        sql_ok = _compare_sql(rows_pred, rows_gold)
                    except Exception:
                        sql_ok = 0
                else:
                    sql_ok = 0

            else:
                # eval_type == "text" → hoy medimos grounding/hallucination únicamente
                pass

        rows_out.append({
            "id": it["id"], "provider": "gemini", "model": MODEL, "lang": it.get("lang","es"),
            "latency_ms": latency_ms, "in_tokens": in_tok, "out_tokens": out_tok,
            "json_ok": json_valid, "numeric_acc": numeric_acc, "numeric_mae": numeric_mae,
            "sql_ok": sql_ok, "grounded": grounded, "hallucination": halluc,
            "raw": raw, "json_error": err
        })

    # ====== Salidas ======
    df = pd.DataFrame(rows_out)
    out_dir = Path(__file__).parent.parent / "runs"
    out_dir.mkdir(exist_ok=True)
    detail_path = out_dir / "per_question.csv"
    summary_path = out_dir / "summary.csv"
    df.to_csv(detail_path, index=False, encoding="utf-8")

    def _safe_mean(series):
        s = [x for x in series if isinstance(x, (int, float)) and x is not None]
        return (sum(s) / len(s)) if s else None

    summary = (
        df.groupby(["provider", "model", "lang"])
          .agg(n=("id", "count"),
               json_ok=("json_ok", "mean"),
               numeric_acc=("numeric_acc", _safe_mean),
               sql_ok=("sql_ok", _safe_mean),
               grounded=("grounded", "mean"),
               hallucination=("hallucination", "mean"),
               p50_latency=("latency_ms", lambda s: float(pd.Series(s).quantile(0.5))),
               p95_latency=("latency_ms", lambda s: float(pd.Series(s).quantile(0.95))))
          .reset_index()
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    return {
        "detail_path": str(detail_path),
        "summary_path": str(summary_path),
        "summary": summary.to_dict(orient="records"),
    }

# ====== Modo script (opcional) ======
if __name__ == "__main__":
    from pprint import pprint
    pprint(run_eval())
