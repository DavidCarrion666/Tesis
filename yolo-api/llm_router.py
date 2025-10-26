# llm_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from database import SessionLocal
from models import Detection, Video
from datetime import datetime
from math import sqrt
import json

router = APIRouter(prefix="/llm", tags=["LLM (heurístico)"])

# ========= Schemas para el endpoint /llm/ask =========
class AskBody(BaseModel):
    question: str
    video_id: str  # UUID en string
    provider: Optional[str] = "heuristic"  # placeholder (openai/gemini/deepseek en el futuro)


# ========= Helpers internos =========
def _safe_class_name(object_class: Optional[str]) -> str:
    """
    Tus detecciones guardan object_class como 'Car 0.87'.
    Aquí nos quedamos con el nombre de clase a la izquierda.
    """
    if not object_class:
        return "unk"
    return object_class.split()[0]


# ========= Funciones de analítica (SE USAN COMO TOOLS CON EL LLM) =========
def get_summary(db: Session, video_id: str) -> Dict[str, Any]:
    """
    Resumen general: conteo por clase, tracks únicos, fps, duración.
    """
    q = (
        db.query(Detection.object_class, Detection.track_id)
          .filter(Detection.video_id == video_id)
    )
    rows = q.all()
    by_class: Dict[str, int] = {}
    unique_tracks: set[int] = set()

    for oclass, tid in rows:
        klass = _safe_class_name(oclass)
        by_class[klass] = by_class.get(klass, 0) + 1
        if tid is not None:
            unique_tracks.add(tid)

    video = db.query(Video).filter(Video.id == video_id).first()
    fps = getattr(video, "fps", None)
    duration_s = getattr(video, "duration_s", None)

    return {
        "detections_by_class": by_class,
        "unique_tracks": len(unique_tracks),
        "fps": fps,
        "duration_s": duration_s,
    }


def get_colors(db: Session, video_id: str) -> Dict[str, Dict[str, int]]:
    """
    Distribución de colores por clase. Lee extra.color_name (JSONB).
    """
    q = (
        db.query(Detection.object_class, Detection.extra)
          .filter(Detection.video_id == video_id)
    )
    acc: Dict[str, Dict[str, int]] = {}

    for oclass, extra in q.all():
        klass = _safe_class_name(oclass)

        color = ""
        if isinstance(extra, dict):
            color = (extra.get("color_name") or "").strip()
        else:
            # Si el driver devolviera JSON como str, intentamos parsear:
            try:
                color = (json.loads(extra).get("color_name") or "").strip()
            except Exception:
                color = ""

        if not color:
            color = "desconocido"

        if klass not in acc:
            acc[klass] = {}
        acc[klass][color] = acc[klass].get(color, 0) + 1

    return acc


def get_tracks(db: Session, video_id: str, klass: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Lista de tracks con duración en frames, opcionalmente filtrados por clase.
    """
    q = (
        db.query(Detection.track_id, Detection.object_class, Detection.frame_number)
          .filter(Detection.video_id == video_id, Detection.track_id.isnot(None))
    )
    rows = q.all()

    tracks: Dict[int, Dict[str, Any]] = {}
    for tid, oclass, fr in rows:
        if tid is None:
            continue
        name = _safe_class_name(oclass)
        if klass and name.lower() != klass.lower():
            continue
        tr = tracks.setdefault(tid, {"track_id": tid, "class": name, "frames": []})
        tr["frames"].append(fr)

    # Resumen por track
    out: List[Dict[str, Any]] = []
    for tid, tr in tracks.items():
        tr["frames"].sort()
        out.append({
            "track_id": tid,
            "class": tr["class"],
            "length_frames": len(tr["frames"]),
            "first_frame": tr["frames"][0],
            "last_frame": tr["frames"][-1],
        })

    # Ordenamos por duración descendente
    out.sort(key=lambda x: x["length_frames"], reverse=True)
    return out


def get_trajectory(db: Session, video_id: str, track_id: int) -> List[Dict[str, Any]]:
    """
    Trayectoria (centroide) de un track: lista de {frame, ts, cx, cy}.
    """
    q = (
        db.query(
            Detection.frame_number, Detection.ts,
            Detection.x1, Detection.y1, Detection.x2, Detection.y2
        )
        .filter(Detection.video_id == video_id, Detection.track_id == track_id)
        .order_by(Detection.frame_number.asc())
    )
    traj: List[Dict[str, Any]] = []
    for fr, ts, x1, y1, x2, y2 in q.all():
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        traj.append({"frame": fr, "ts": ts, "cx": cx, "cy": cy})
    return traj


def get_speed(db: Session, video_id: str, track_id: int, fps_fallback: float = 30.0) -> Dict[str, Any]:
    """
    Velocidad media y pico en píxeles/segundo del track (derivada de trayectoria).
    Nota: si calibras escala y ángulo podrás convertir a km/h.
    """
    video = db.query(Video).filter(Video.id == video_id).first()
    fps = getattr(video, "fps", None) or fps_fallback

    traj = get_trajectory(db, video_id, track_id)
    if len(traj) < 2:
        return {"track_id": track_id, "fps": fps, "avg_px_s": 0.0, "max_px_s": 0.0, "samples": 0}

    speeds = []
    prev = traj[0]
    for cur in traj[1:]:
        d = sqrt((cur["cx"] - prev["cx"])**2 + (cur["cy"] - prev["cy"])**2)
        v = d * fps  # píxeles por segundo (Δframe = 1/fps)
        speeds.append(v)
        prev = cur

    return {
        "track_id": track_id,
        "fps": fps,
        "avg_px_s": sum(speeds) / len(speeds),
        "max_px_s": max(speeds),
        "samples": len(speeds),
    }


# ========= “Catálogo” de herramientas (para referencia) =========
TOOLS = {
    "get_summary": {
        "description": "Resumen general del video.",
        "args": ["video_id"],
        "fn": lambda db, args: get_summary(db, args["video_id"]),
    },
    "get_colors": {
        "description": "Distribución de colores por clase.",
        "args": ["video_id"],
        "fn": lambda db, args: get_colors(db, args["video_id"]),
    },
    "get_tracks": {
        "description": "Tracks y su duración (frames).",
        "args": ["video_id", "klass?"],
        "fn": lambda db, args: get_tracks(db, args["video_id"], args.get("klass")),
    },
    "get_trajectory": {
        "description": "Trayectoria (cx, cy) para un track.",
        "args": ["video_id", "track_id"],
        "fn": lambda db, args: get_trajectory(db, args["video_id"], int(args["track_id"])),
    },
    "get_speed": {
        "description": "Velocidad media y pico (px/s) de un track.",
        "args": ["video_id", "track_id"],
        "fn": lambda db, args: get_speed(db, args["video_id"], int(args["track_id"])),
    },
}


# ========= Endpoint heurístico (útil sin LLM real) =========
@router.post("/ask")
def ask_llm(body: AskBody):
    """
    Ruta mínima que ya consulta tu BD SIN proveedor LLM.
    Usa heurísticas simples para decidir qué función de analítica llamar.
    Cuando conectes Gemini u otro, reutiliza estas funciones como 'tools'.
    """
    db = SessionLocal()
    try:
        q = body.question.lower()

        if "color" in q:
            data = TOOLS["get_colors"]["fn"](db, {"video_id": body.video_id})
            answer = f"Distribución de colores por clase: {data}"

        elif any(k in q for k in ["velocidad", "rápido", "rapido"]):
            tracks = TOOLS["get_tracks"]["fn"](db, {"video_id": body.video_id})
            if not tracks:
                answer = "No hay tracks para calcular velocidad."
            else:
                top = tracks[0]
                speed = TOOLS["get_speed"]["fn"](
                    db, {"video_id": body.video_id, "track_id": top["track_id"]}
                )
                answer = (
                    f"Track #{speed['track_id']}: vel media ~{speed['avg_px_s']:.1f}px/s, "
                    f"pico ~{speed['max_px_s']:.1f}px/s (fps={speed['fps']})."
                )

        elif any(k in q for k in ["trayector", "recorrido", "ruta"]):
            # si el usuario menciona un id (e.g., "track 5")
            import re
            m = re.search(r"(track|id)\s*#?\s*(\d+)", q)
            if m:
                tid = int(m.group(2))
            else:
                tracks = TOOLS["get_tracks"]["fn"](db, {"video_id": body.video_id})
                if not tracks:
                    return {"answer": "No hay tracks.", "tool_calls": []}
                tid = tracks[0]["track_id"]

            traj = TOOLS["get_trajectory"]["fn"](db, {"video_id": body.video_id, "track_id": tid})
            answer = f"Trayectoria del track #{tid} con {len(traj)} puntos. Ejemplo primeros 5: {traj[:5]}"

        elif any(k in q for k in ["cuántos", "cuantos", "conteo", "resumen", "summary"]):
            data = TOOLS["get_summary"]["fn"](db, {"video_id": body.video_id})
            answer = (
                f"Resumen: {data['detections_by_class']} | "
                f"tracks únicos: {data['unique_tracks']} | fps={data['fps']}."
            )

        else:
            # por defecto devuelve resumen corto
            data = TOOLS["get_summary"]["fn"](db, {"video_id": body.video_id})
            answer = f"[Default] {data['detections_by_class']} (tracks únicos: {data['unique_tracks']})."

        return {"answer": answer, "tool_calls": "heuristic"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
