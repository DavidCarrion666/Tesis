from fastapi import FastAPI, UploadFile, File, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Video, Detection
import os
import time
import cv2
import base64
import asyncio
import json
from typing import Optional
from uuid import UUID
from datetime import datetime, timezone

# ======== Helpers de color (LAB + HSV) ========
import numpy as np

HUE_RANGES = [
    ("rojo",     (0, 12)),
    ("naranja",  (12, 22)),
    ("amarillo", (22, 35)),
    ("verde",    (35, 85)),
    ("cian",     (85, 100)),
    ("azul",     (100, 130)),
    ("morado",   (130, 155)),
    ("rosa",     (155, 178)),
    ("rojo",     (178, 180)),  # wrap
]

def _name_from_h(h: int) -> str:
    for name, (lo, hi) in HUE_RANGES:
        if lo <= h < hi:
            return name
    return "desconocido"

def vehicle_color_name(frame_bgr, x1, y1, x2, y2):
    """
    Clasifica color de vehículo combinando LAB (para acromáticos) + HSV (para cromáticos).
    Devuelve (nombre, (b,g,r)).
    """
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

    # Acromáticos primero
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

    # Cromáticos por HSV
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
# =============================================

# ---- Routers LLM (heurístico y Gemini) ----
from llm_router import router as llm_router            # usa funciones locales sin LLM real
from llm_gemini_router import router as gemini_router # usa Gemini con function-calling

# Crear app FastAPI
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registro de routers LLM
app.include_router(llm_router)
app.include_router(gemini_router)

# Modelo YOLO
model = YOLO("best.pt")

# Carpeta de subidas
os.makedirs("uploads", exist_ok=True)

# --------------------- SUBIDA DE VIDEO ---------------------
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Sube el video y lo guarda en la base de datos"""
    db: Session = SessionLocal()
    try:
        ts = int(time.time())
        filename = f"{ts}_{file.filename}"
        filepath = os.path.join("uploads", filename)

        with open(filepath, "wb") as f:
            f.write(await file.read())

        db_video = Video(
            video_name=filename,
            source="drone_01",
            fps=30,
            duration_s=3600,
        )
        db.add(db_video)
        db.commit()
        db.refresh(db_video)

        return {"filename": filepath, "video_id": db_video.id}
    finally:
        db.close()

# --------------------- STREAM EN TIEMPO REAL ---------------------
@app.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    file: Optional[str] = Query(None),
    video_id: Optional[UUID] = Query(None),
):
    """
    Envía frames con YOLO + tracking y guarda detecciones (incluye track_id y color) en DB.
    """
    await websocket.accept()

    if not file:
        await websocket.send_text("ERROR: Falta parámetro 'file'")
        await websocket.close()
        return

    if video_id is None:
        video_id = 0  # mantiene compatibilidad con tu versión original

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

            # Detección + Tracking (ByteTrack)
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

                    # Color del vehículo
                    color_name, bgr_mean = vehicle_color_name(frame, x1, y1, x2, y2)

                    # Guardar en DB
                    det = Detection(
                        video_id=video_id,
                        frame_number=frame_idx,
                        ts=datetime.fromtimestamp(time.time(), tz=timezone.utc),
                        object_class=label,
                        confidence=conf,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        track_id=track_id,
                        extra=json.dumps({
                            "color_name": color_name,
                            "bgr_mean": bgr_mean
                        }),
                    )
                    db.add(det)
                    db.commit()

                    # Dibujo
                    draw_label = f"{label}" + (f" #{track_id}" if track_id is not None else "")
                    draw_label = f"{draw_label} | {color_name}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, draw_label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                await asyncio.sleep(0.03)
                continue

            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            try:
                await websocket.send_text(jpg_as_text)
            except Exception:
                break

            await asyncio.sleep(0.03)

        cap.release()
        try:
            await websocket.send_text("<<EOF>>")
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass

    except Exception as e:
        try:
            await websocket.send_text(f"ERROR: {str(e)}")
            await websocket.close()
        except Exception:
            pass
    finally:
        db.close()
