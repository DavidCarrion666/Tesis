from fastapi import APIRouter
from pydantic import BaseModel
from .engine import ask_gemini, run_sql


router = APIRouter(prefix="/llm-text")


class LLMQuery(BaseModel):
    question: str
    video_id: str


@router.post("/query")
def query_llm_text(payload: LLMQuery):
    result = ask_gemini(
        question=payload.question,
        video_id=payload.video_id
    )

    if result.get("error"):
        return {
            "response": "Hubo un problema procesando tu pregunta.",
            "detalle": result
        }

    answer = result["answer"]
    sql = result["sql_used"]
    t = result["type"]

    if not sql:
        return { "response": answer }

    # Normalizar SQL mínimo
    sql = sql.replace("vehicle", "car")

    rows = run_sql(sql)

    if isinstance(rows, dict) and "error" in rows:
        return {
            "response": f"No pude ejecutar la consulta SQL: {rows['error']}",
            "sql_used": sql
        }

    # Caso valores escalares
    if len(rows) == 1 and len(rows[0]) == 1:
        real_value = rows[0][0]
        return {
            "response": f"{answer} {real_value}.",
            "sql_used": sql
        }

    # Caso tablas o múltiples filas
    pretty_rows = [str(list(r)) for r in rows]
    result_str = "; ".join(pretty_rows)

    return {
        "response": f"{answer} Resultados: {result_str}.",
        "sql_used": sql
    }
