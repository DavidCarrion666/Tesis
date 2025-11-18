import json
import os
from pathlib import Path
from jsonschema import validate, ValidationError
import google.generativeai as genai

from sqlalchemy import text
from database import engine as db_engine

from .prompts import SYSTEM_PROMPT_ES


# =============================
#  CONFIGURACIÓN GEMINI
# =============================

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Falta GOOGLE_API_KEY en variables de entorno")

genai.configure(api_key=API_KEY)

SCHEMA_PATH = Path(__file__).with_name("schema.json")
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


# =============================
#  EJECUCIÓN DE SQL
# =============================

def run_sql(sql: str):
    """
    Ejecuta SQL directamente contra la base detections_norm.
    Devuelve lista de filas o dict con error.
    """
    try:
        with db_engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            return rows
    except Exception as e:
        return {"error": str(e)}


# =============================
#  LLAMADA A GEMINI
# =============================

def ask_gemini(question: str, video_id: str):
    """
    Llama a Gemini 2.0 con JSON estricto usando SYSTEM_PROMPT_ES.
    Devuelve un dict ya validado contra schema.json
    """

    user_prompt = f"""
Pregunta: {question}
video_id: {video_id}

Responde SOLO con JSON válido según el esquema. No agregues texto fuera del JSON.
"""

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT_ES
    )

    resp = model.generate_content(
        [user_prompt],
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.0
        }
    )

    raw = resp.text or ""

    # Intentar parsear JSON
    try:
        obj = json.loads(raw)
    except Exception as e:
        return {
            "error": "JSON inválido",
            "raw": raw,
            "exception": str(e)
        }

    # Validar con schema
    try:
        validate(obj, SCHEMA)
    except ValidationError as ve:
        return {
            "error": "No cumple el schema",
            "raw": raw,
            "exception": str(ve)
        }

    return obj
