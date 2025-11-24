from fastapi import FastAPI, WebSocket, Query, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ultralytics import YOLO
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Video, Detection

import os
import cv2
import base64
import asyncio
import json
from typing import Optional
from uuid import UUID
from datetime import datetime, timezone
import numpy as np

from uuid import uuid4

import google.generativeai as genai

# ============================================================
#   CONFIGURACIÓN GEMINI
# ============================================================
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("ERROR: No existe variable GOOGLE_API_KEY en el entorno.")

genai.configure(api_key=API_KEY)

# ============================================================
#   HELPERS DE COLOR
# ============================================================
HUE_RANGES = [
    ("rojo",     (0, 12)),
    ("naranja",  (12, 22)),
    ("amarillo", (22, 35)),
    ("verde",    (35, 85)),
    ("cian",     (85, 100)),
    ("azul",     (100, 130)),
    ("morado",   (130, 155)),
    ("rosa",     (155, 178)),
    ("rojo",     (178, 180)),
]


def _name_from_h(h: int) -> str:
    for name, (lo, hi) in HUE_RANGES:
        if lo <= h < hi:
            return name
    return "desconocido"


def vehicle_color_name(frame_bgr, x1, y1, x2, y2):
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return "desconocido", (0, 0, 0)

    small = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
    bgr_mean = tuple(int(x) for x in small.reshape(-1, 3).mean(axis=0))

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b)

    Lp = np.percentile(L, 50)
    Cp = np.percentile(chroma, 50)

    if Lp < 38 or (Lp < 52 and Cp < 12):
        return "negro", bgr_mean
    if Lp > 86 and Cp < 10:
        return "blanco", bgr_mean
    if Cp < 14:
        if Lp < 52:
            return "gris oscuro", bgr_mean
        elif Lp < 70:
            return "gris", bgr_mean
        else:
            return "gris claro", bgr_mean

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int32)
    S = hsv[:, :, 1].astype(np.int32)
    V = hsv[:, :, 2].astype(np.int32)

    mask = V > 40
    if mask.sum() < 50:
        return "negro", bgr_mean

    h_med = int(np.median(H[mask]))
    s_med = int(np.median(S[mask]))
    v_med = int(np.median(V[mask]))

    base = _name_from_h(h_med)
    if v_med < 80 and s_med > 40:
        return f"{base} oscuro", bgr_mean
    if v_med > 170 and s_med < 160:
        return f"{base} claro", bgr_mean
    return base, bgr_mean


# ============================================================
#   APP FASTAPI
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#   YOLO
# ============================================================
model = YOLO("best.pt")
os.makedirs("uploads", exist_ok=True)


# ============================================================
#   MODELOS Pydantic
# ============================================================
class VideoQuery(BaseModel):
    video_id: UUID
    question: str
    step: int = 30   # cada cuántos frames muestrear para representar el video


# ============================================================
#   ENDPOINT /upload (SUBIR VIDEO Y REGISTRARLO)
# ============================================================
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Sube un video, lo guarda en /uploads y lo registra en la base de datos.
    Devuelve video_id para futura consulta multimodal.
    """
    db: Session = SessionLocal()

    try:
        ext = os.path.splitext(file.filename)[1]
        filename = f"{int(datetime.now().timestamp())}_{uuid4()}{ext}"
        save_path = os.path.join("uploads", filename)

        with open(save_path, "wb") as f:
            f.write(await file.read())

        # Ajusta estos campos a tu modelo real de Video
        video = Video(
            video_name=filename,
            source="upload",
            fps=30,
            duration_s=0,
        )
        db.add(video)
        db.commit()
        db.refresh(video)

        return {
            "video_id": str(video.id),
            "video_name": filename,
            "path": save_path,
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        db.close()


# ============================================================
#   HELPERS BÁSICOS
# ============================================================
def get_video_filepath(db: Session, video_id: UUID) -> str:
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise ValueError("Video no encontrado")
    return os.path.join("uploads", video.video_name)


# ============================================================
#   REPRESENTATIVE FRAMES (1 frame cada step)
# ============================================================
def get_representative_frames(db: Session, video_id: UUID, step: int = 30):
    """
    Extrae frames representativos del video.
    step = cada cuántos frames muestrear (default = 30)
    """
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise ValueError("Video no encontrado")

    video_path = os.path.join("uploads", video.video_name)

    detections = (
        db.query(Detection)
        .filter(Detection.video_id == video_id)
        .order_by(Detection.frame_number.asc())
        .all()
    )
    if not detections:
        raise ValueError("No existen detecciones para este video")

    detected_frames = sorted({d.frame_number for d in detections})

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video")

    representative = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for f in range(0, total_frames, step):
        if f not in detected_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            continue

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_dets = [
            {
                "object_class": d.object_class,
                "confidence": d.confidence,
                "bbox": {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2},
                "track_id": d.track_id,
                "extra": json.loads(d.extra) if d.extra else None,
            }
            for d in detections if d.frame_number == f
        ]

        representative.append({
            "frame_number": f,
            "image_bytes": buffer.tobytes(),
            "detections": frame_dets,
        })

    cap.release()
    return representative


# ============================================================
#   GLOBAL VIDEO SUMMARY (ANÁLISIS GENERAL DEL VIDEO)
# ============================================================
def build_global_video_summary(db: Session, video_id: UUID):
    """
    Lee TODAS las detecciones del video y genera un resumen global:
    - vehículos únicos (track_id)
    - colores globales
    - conteo por minuto
    - posibles colisiones simples por IoU
    """
    detections = (
        db.query(Detection)
        .filter(Detection.video_id == video_id)
        .order_by(Detection.frame_number.asc())
        .all()
    )

    if not detections:
        raise ValueError("No existen detecciones para este video")

    frames = sorted({d.frame_number for d in detections})
    max_frame = frames[-1]

    vehicle_tracks = {}
    color_stats = {}

    for d in detections:
        tid = d.track_id
        if tid is None:
            continue

        if tid not in vehicle_tracks:
            vehicle_tracks[tid] = {
                "frames": [],
                "colors": [],
                "confidences": [],
            }

        vehicle_tracks[tid]["frames"].append(d.frame_number)

        try:
            extra_data = json.loads(d.extra) if d.extra else {}
        except Exception:
            extra_data = {}

        color_name = extra_data.get("color_name")
        if color_name:
            vehicle_tracks[tid]["colors"].append(color_name)
            color_stats.setdefault(color_name, 0)
            color_stats[color_name] += 1

        vehicle_tracks[tid]["confidences"].append(d.confidence)

    vehicles_summary = {}
    for tid, info in vehicle_tracks.items():
        if info["colors"]:
            dominant_color = max(set(info["colors"]), key=info["colors"].count)
        else:
            dominant_color = "desconocido"

        avg_conf = float(np.mean(info["confidences"])) if info["confidences"] else 0.0

        vehicles_summary[tid] = {
            "frames": info["frames"],
            "color": dominant_color,
            "avg_confidence": avg_conf,
        }

    traffic_by_minute = {}
    for d in detections:
        minute = d.frame_number // (30 * 60)  # asumiendo 30 FPS
        traffic_by_minute.setdefault(minute, 0)
        traffic_by_minute[minute] += 1

    possible_collision_frames = []

    def iou(b1, b2):
        x1 = max(b1.x1, b2.x1)
        y1 = max(b1.y1, b2.y1)
        x2 = min(b1.x2, b2.x2)
        y2 = min(b1.y2, b2.y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1)
        area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    for f in frames:
        frame_boxes = [d for d in detections if d.frame_number == f]
        for i in range(len(frame_boxes)):
            for j in range(i + 1, len(frame_boxes)):
                if iou(frame_boxes[i], frame_boxes[j]) > 0.25:
                    possible_collision_frames.append(f)

    summary = {
        "video_id": str(video_id),
        "total_frames": max_frame,
        "vehicles_count": len(vehicles_summary),
        "vehicle_ids": list(vehicles_summary.keys()),
        "vehicles": vehicles_summary,
        "color_stats": color_stats,
        "traffic_by_minute": traffic_by_minute,
        "possible_events": {
            "collision_frames": sorted(set(possible_collision_frames))
        },
    }

    return summary


# ============================================================
#   LLAMADA MULTIMODAL GEMINI (VIDEO COMPLETO)
# ============================================================
def call_multimodal_video_llm(
    representative_frames: list,
    global_summary: dict,
    question: str,
):
    """
    Envía a Gemini:
    - Resumen global del video (JSON)
    - Varios frames representativos + sus detecciones
    - Pregunta general
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""
Eres un analista de tráfico MULTIMODAL a nivel de VIDEO COMPLETO.

Recibirás:
1) Un RESUMEN GLOBAL del video en JSON (vehículos, colores, tráfico por minuto, posibles colisiones).
2) Varias IMÁGENES representativas de diferentes momentos del video, cada una con sus detecciones.
3) Una PREGUNTA general sobre TODO el video.

Tu tarea:
- Combinar la información VISUAL de las imágenes con la información ESTRUCTURADA del JSON global.
- Contestar la pregunta SIEMPRE en función de todo el video, no solo de una imagen.
- Responder exclusivamente en JSON con el siguiente formato:

{{
  "answer": string,
  "type": "numeric" | "text" | "count" | "event",
  "value": number | string | null,
  "evidence": string[],
  "confidence": number
}}

Pregunta del usuario: "{question}"
        """

        contents = [prompt]

        contents.append(json.dumps(
            {"global_summary": global_summary},
            ensure_ascii=False
        ))

        for rf in representative_frames:
            contents.append({
                "mime_type": "image/jpeg",
                "data": rf["image_bytes"],
            })
            contents.append(json.dumps({
                "frame_number": rf["frame_number"],
                "detections": rf["detections"],
            }, ensure_ascii=False))

        response = model.generate_content(contents=contents)
        raw_text = response.text.strip()

        try:
            parsed = json.loads(raw_text)
            return parsed
        except Exception:
            return {"raw_response": raw_text}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
#   ENDPOINT MULTIMODAL GLOBAL DEL VIDEO
# ============================================================
@app.post("/multimodal/video-query")
async def multimodal_video_query(payload: VideoQuery = Body(...)):
    db: Session = SessionLocal()
    try:
        frames = get_representative_frames(db, payload.video_id, payload.step)
        if not frames:
            raise ValueError("No se pudieron obtener frames representativos")

        global_summary = build_global_video_summary(db, payload.video_id)

        llm_resp = call_multimodal_video_llm(
            representative_frames=frames,
            global_summary=global_summary,
            question=payload.question,
        )

        return {
            "video_id": str(payload.video_id),
            "question": payload.question,
            "used_frames": [f["frame_number"] for f in frames],
            "global_summary": global_summary,
            "llm_response": llm_resp,
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        db.close()


# ============================================================
#   WEBSOCKET: STREAM + DETECCIÓN + GUARDADO EN DB
# ============================================================
@app.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    file: Optional[str] = Query(None),
    video_id: Optional[UUID] = Query(None),
):
    await websocket.accept()

    if not file:
        await websocket.send_text("ERROR: Falta parámetro 'file'")
        await websocket.close()
        return

    if video_id is None:
        video_id = 0  # ajusta si quieres forzar UUID real

    db: Session = SessionLocal()
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        await websocket.send_text("ERROR: No se puede abrir el video")
        await websocket.close()
        db.close()
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            res = results[0]

            if hasattr(res, "boxes") and len(res.boxes) > 0:
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                for box in res.boxes:
                    xyxy = box.xyxy[0].tolist()
                    cls = int(box.cls)
                    conf = float(box.conf)
                    x1, y1, x2, y2 = map(int, xyxy)

                    track_id = None
                    if hasattr(box, "id") and box.id is not None:
                        try:
                            track_id = int(box.id[0])
                        except Exception:
                            track_id = None

                    class_name = model.names[cls] if hasattr(model, "names") else str(cls)
                    label = f"{class_name} {conf:.2f}"

                    color_name, bgr_mean = vehicle_color_name(
                        frame, x1, y1, x2, y2
                    )

                    det = Detection(
                        video_id=video_id,
                        frame_number=frame_idx,
                        ts=datetime.now(timezone.utc),
                        object_class=label,
                        confidence=conf,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        track_id=track_id,
                        extra=json.dumps({
                            "color_name": color_name,
                            "bgr_mean": bgr_mean,
                        }),
                    )
                    db.add(det)
                    db.commit()

                    draw_label = f"{label}"
                    if track_id:
                        draw_label += f" #{track_id}"
                    draw_label += f" | {color_name}"

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        draw_label,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                await asyncio.sleep(0.03)
                continue

            await websocket.send_text(
                base64.b64encode(buffer).decode("utf-8")
            )
            await asyncio.sleep(0.03)

        cap.release()
        try:
            await websocket.send_text("<<EOF>>")
            await websocket.close()
        except Exception:
            pass

    finally:
        db.close()
