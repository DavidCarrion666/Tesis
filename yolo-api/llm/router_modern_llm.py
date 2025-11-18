from fastapi import APIRouter
from pydantic import BaseModel
from .engine import ask_gemini


router = APIRouter(prefix="/llm-modern")


class LLMQuery(BaseModel):
    question: str
    video_id: str


@router.post("/query")
def query_llm(payload: LLMQuery):
    """
    Endpoint moderno:
    Devuelve JSON válido según el esquema del sistema LLM.
    """
    result = ask_gemini(
        question=payload.question,
        video_id=payload.video_id
    )
    return result
