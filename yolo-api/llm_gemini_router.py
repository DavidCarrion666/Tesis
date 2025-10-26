# llm_gemini_router.py
from typing import Any, Dict, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from database import SessionLocal
# IMPORTA tus funciones de métricas reales:
from metrics_tools import (
    get_summary,
    get_colors,
    get_tracks,
    get_trajectory,
    get_speed,
)

from providers_gemini import gemini_plan_and_answer

router = APIRouter(prefix="/llm-gemini", tags=["llm-gemini"])

class AskBody(BaseModel):
    question: str
    video_id: Optional[str] = None

@router.post("/ask")
def ask(body: AskBody):
    # 1) Chequeo mínimo
    if not body.video_id:
        return {"answer": "Primero sube un video (no hay video_id)."}

    db = SessionLocal()

    def exec_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta las tools contra la base. NO asumir que el LLM mandó video_id ni tipos correctos.
        """
        try:
            vid = args.get("video_id") or body.video_id  # fallback a la que vino en el POST
            if not vid:
                return {"error": "video_id es requerido"}

            klass = args.get("klass")
            tid = args.get("track_id")

            if name == "get_summary":
                return get_summary(db, vid)
            if name == "get_colors":
                return get_colors(db, vid)
            if name == "get_tracks":
                return get_tracks(db, vid, klass)
            if name == "get_trajectory":
                if tid is None:
                    return {"error": "track_id es requerido"}
                return get_trajectory(db, vid, int(tid))
            if name == "get_speed":
                if tid is None:
                    return {"error": "track_id es requerido"}
                return get_speed(db, vid, int(tid))

            return {"error": f"tool '{name}' no implementada"}

        except Exception as e:
            return {"error": f"tool '{name}' lanzó excepción: {e}"}

    try:
        result = gemini_plan_and_answer(
            question=body.question,
            video_id=body.video_id,
            tool_executor=exec_tool
        )
        return result
    finally:
        db.close()
