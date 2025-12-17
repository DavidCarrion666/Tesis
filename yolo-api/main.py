from fastapi import FastAPI, WebSocket, Query, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Video, Detection, VideoDocument
import os
import time
import cv2
import base64
import asyncio
import json
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime, timezone
import torch
import numpy as np
from pydantic import BaseModel

from uuid import uuid4
import math
from openai import OpenAI

# ⚠️ Solo dejo heuristic_llm_router si realmente lo usas.
from llm_router import router as heuristic_llm_router
# from llm_gemini_router import router as gemini_llm_router   # ⬅️ ELIMINADO


# ============================================================
#   CONFIGURACIÓN OPENAI (CHATGPT)
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("ERROR: No existe variable OPENAI_API_KEY en el entorno.")

client = OpenAI(api_key=OPENAI_API_KEY)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


# ============================================================
#   EMBEDDINGS CON OPENAI (VECTORIAL)
# ============================================================
def embed_text(text: str) -> List[float]:
    """
    Convierte un texto en un vector (lista de floats) usando OpenAI embeddings.
    """
    if not text or not text.strip():
        return []

    resp = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ============================================================
# === CONFIGURACIÓN CUDA / MODELO YOLO ===
# ============================================================
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

model = YOLO("best.pt")
model.to("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# === CLASIFICADOR DE COLOR VEHICULAR (KMEANS + LAB + HSV) ===
# ============================================================
HUE_RANGES = [
    ("rojo", (0, 12)),
    ("naranja", (12, 22)),
    ("amarillo", (22, 35)),
    ("verde", (35, 85)),
    ("cian", (85, 100)),
    ("azul", (100, 130)),
    ("morado", (130, 155)),
    ("rosa", (155, 178)),
    ("rojo", (178, 180)),
]


def _name_from_h(h: int) -> str:
    for name, (lo, hi) in HUE_RANGES:
        if lo <= h < hi:
            return name
    return "desconocido"


def vehicle_color_name(frame_bgr, x1, y1, x2, y2):
    """
    Clasifica el color de un vehículo de forma más robusta y pesada:
    - Recorta el centro del bbox para evitar fondo.
    - Aplica K-Means en espacio Lab para obtener el color dominante.
    - Decide si es neutro (negro / gris / blanco) o cromático.
    - Si es cromático, usa HSV para mapear a un nombre de color.

    Devuelve: (nombre_color, bgr_mean_tuple)
    """
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    patch = frame_bgr[y1:y2, x1:x2]

    if patch.size == 0:
        return "desconocido", (0, 0, 0)

    # 1) Usar SOLO la parte central del bbox para minimizar fondo
    ph, pw = patch.shape[:2]
    cx1 = int(pw * 0.2)
    cx2 = int(pw * 0.8)
    cy1 = int(ph * 0.2)
    cy2 = int(ph * 0.8)
    core = patch[cy1:cy2, cx1:cx2]
    if core.size == 0:
        core = patch

    # 2) Reducir tamaño para que K-Means no explote
    small = cv2.resize(core, (64, 64), interpolation=cv2.INTER_AREA)

    # 3) Convertir a Lab para separar brillo vs cromaticidad
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)  # L, a, b en [0..255]
    Z = lab.reshape((-1, 3)).astype(np.float32)

    # 4) K-Means para encontrar 3 colores dominantes
    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    try:
        compactness, labels, centers = cv2.kmeans(
            Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
    except Exception:
        # Si K-Means falla, usar promedio simple
        bgr_mean_fallback = tuple(int(x) for x in small.reshape(-1, 3).mean(axis=0))
        return "desconocido", bgr_mean_fallback

    # 5) Cluster dominante (el que más píxeles tiene)
    labels = labels.flatten()
    counts = np.bincount(labels)
    dom_idx = int(np.argmax(counts))
    dom_lab = centers[dom_idx]  # [L, a, b] en float32

    # 6) Convertir centro dominante a BGR y HSV
    dom_lab_img = np.uint8([[dom_lab]])
    dom_bgr = cv2.cvtColor(dom_lab_img, cv2.COLOR_LAB2BGR)[0, 0, :]
    dom_hsv = cv2.cvtColor(np.uint8([[dom_bgr]]), cv2.COLOR_BGR2HSV)[0, 0, :]
    H, S, V = int(dom_hsv[0]), int(dom_hsv[1]), int(dom_hsv[2])

    # BGR medio del core (para guardar en BD / debug)
    bgr_mean = tuple(int(x) for x in core.reshape(-1, 3).mean(axis=0))

    # 7) Decidir si es NEUTRO (blanco/gris/negro) o CROMÁTICO
    # proporción de píxeles poco saturados en el parche
    hsv_full = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    S_full = hsv_full[:, :, 1]
    V_full = hsv_full[:, :, 2]
    low_sat_mask = S_full < 40  # umbral de saturación
    low_sat_ratio = float(low_sat_mask.sum()) / float(low_sat_mask.size)

    # Caso NEUTRO: mayoría de píxeles poco saturados
    if low_sat_ratio > 0.6 or S < 40:
        # Clasificar por brillo (V) en la imagen completa
        v_med = int(np.median(V_full))

        if v_med < 60:
            return "negro", bgr_mean
        if v_med > 200:
            return "blanco", bgr_mean
        if v_med < 120:
            return "gris oscuro", bgr_mean
        if v_med < 180:
            return "gris", bgr_mean
        return "gris claro", bgr_mean

    # Caso CROMÁTICO: suficiente saturación → usamos H
    base = _name_from_h(H)

    # Afinar con brillo
    if V < 80:
        base = f"{base} oscuro"
    elif V > 190 and S < 160:
        base = f"{base} claro"

    return base, bgr_mean


# ============================================================
# === APP FASTAPI ===
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👉 registra solo las rutas LLM que sí existen
app.include_router(heuristic_llm_router)
# app.include_router(gemini_llm_router)   # ⬅️ ELIMINADO


# ============================================================
#   MODELOS Pydantic
# ============================================================
class VideoQuery(BaseModel):
    video_id: UUID
    question: str
    step: int = 30   # cada cuántos frames muestrear para representar el video


# ============================================================
# === ENDPOINT: UPLOAD VIDEO ===
# ============================================================
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    db = SessionLocal()
    try:
        ts = int(time.time())
        filename = f"{ts}_{file.filename}"
        filepath = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)

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

    vehicle_tracks: Dict[Any, Dict[str, Any]] = {}
    color_stats: Dict[str, int] = {}

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

    traffic_by_minute: Dict[int, int] = {}
    for d in detections:
        minute = d.frame_number // (30 * 60)  # asumiendo 30 FPS
        traffic_by_minute.setdefault(minute, 0)
        traffic_by_minute[minute] += 1

    possible_collision_frames: List[int] = []

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
#   CONSTRUCTORES DE DOCUMENTOS RAG
# ============================================================
def build_vehicle_docs(global_summary: dict) -> List[Dict[str, Any]]:
    """
    Crea documentos a nivel de vehículo a partir del global_summary.
    """
    docs: List[Dict[str, Any]] = []
    vehicles = global_summary.get("vehicles", {})

    for tid, info in vehicles.items():
        frames = info.get("frames", [])
        if frames:
            frame_start = min(frames)
            frame_end = max(frames)
        else:
            frame_start = frame_end = None

        color = info.get("color", "desconocido")
        avg_conf = info.get("avg_confidence", 0.0)

        text = (
            f"Vehículo con track_id {tid}. "
            f"Color predominante: {color}. "
            f"Aparece entre los frames {frame_start} y {frame_end}. "
            f"Confianza promedio de detección: {avg_conf:.2f}."
        )

        docs.append({
            "id": f"veh_{tid}",
            "text": text,
            "metadata": {
                "type": "vehicle",
                "vehicle_id": tid,
                "color": color,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "avg_confidence": avg_conf,
            }
        })

    return docs


def build_time_segment_docs(global_summary: dict) -> List[Dict[str, Any]]:
    """
    Crea documentos a nivel de minuto de video usando traffic_by_minute.
    """
    docs: List[Dict[str, Any]] = []
    tbm = global_summary.get("traffic_by_minute", {})

    for minute, count in tbm.items():
        text = (
            f"En el minuto {minute} del video se registraron "
            f"{count} detecciones de vehículos."
        )

        docs.append({
            "id": f"min_{minute}",
            "text": text,
            "metadata": {
                "type": "minute",
                "minute": minute,
                "traffic_count": count,
            }
        })

    return docs


def build_global_doc(global_summary: dict) -> Dict[str, Any]:
    """
    Documento único con resumen global del video.
    """
    vehicles_count = global_summary.get("vehicles_count", 0)
    color_stats = global_summary.get("color_stats", {})
    possible_events = global_summary.get("possible_events", {})
    collision_frames = possible_events.get("collision_frames", [])

    text = (
        f"Resumen global del video. "
        f"Total de vehículos detectados: {vehicles_count}. "
        f"Distribución de colores: {color_stats}. "
        f"Posibles colisiones en los frames: {collision_frames}."
    )

    return {
        "id": "global_summary",
        "text": text,
        "metadata": {
            "type": "global",
            "vehicles_count": vehicles_count,
            "color_stats": color_stats,
            "collision_frames": collision_frames,
        }
    }


def index_video_knowledge(db: Session, video_id: UUID, global_summary: dict):
    """
    Genera documentos RAG para un video y los guarda en la BD con embeddings.
    """
    # 1) Construir docs a partir del resumen global
    docs: List[Dict[str, Any]] = []
    docs.extend(build_vehicle_docs(global_summary))
    docs.extend(build_time_segment_docs(global_summary))
    docs.append(build_global_doc(global_summary))

    # 2) Borrar docs previos (reindexar limpio si ya existían)
    db.query(VideoDocument).filter(VideoDocument.video_id == video_id).delete()

    # 3) Insertar cada documento con su embedding
    for d in docs:
        emb = embed_text(d["text"])

        video_doc = VideoDocument(
            video_id=video_id,
            source_id=d["id"],
            doc_type=d["metadata"].get("type", "unknown"),
            text=d["text"],
            doc_metadata=d["metadata"],
            embedding=emb,
        )
        db.add(video_doc)

    db.commit()


def rag_retrieve(db: Session, video_id: UUID, question: str, k: int = 8) -> List[Dict[str, Any]]:
    """
    RAG básico: dado un video y una pregunta, devuelve
    los k documentos más similares desde la tabla video_documents.
    """
    docs = db.query(VideoDocument).filter(VideoDocument.video_id == video_id).all()
    if not docs:
        return []

    q_emb = embed_text(question)
    if not q_emb:
        return []

    scored: List[Dict[str, Any]] = []

    for doc in docs:
        emb = doc.embedding or []
        if not emb:
            continue
        score = cosine_similarity(q_emb, emb)
        scored.append({
            "score": float(score),
            "id": str(doc.id),
            "source_id": doc.source_id,
            "doc_type": doc.doc_type,
            "text": doc.text,
            "metadata": doc.doc_metadata,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


# ============================================================
#   LLAMADA CHATGPT (VIDEO COMPLETO + RAG, TEXTO)
# ============================================================
def call_multimodal_video_llm(
    representative_frames: list,
    global_summary: dict,
    question: str,
    retrieved_docs: Optional[List[Dict[str, Any]]] = None,
):
    """
    Llama a ChatGPT (OpenAI) usando:
    - Resumen global del video (global_summary)
    - Documentos RAG recuperados (retrieved_docs)
    - Descripción textual de los frames representativos (detecciones)
    - Pregunta del usuario

    Si el modelo no devuelve JSON válido, se devuelve el texto bruto en 'answer'.
    """
    try:
        # Preparamos texto con los frames representativos
        frames_desc = []
        for rf in representative_frames:
            dets = rf.get("detections", [])
            simple_dets = []
            for d in dets:
                cls = d.get("object_class", "")
                track_id = d.get("track_id")
                extra = d.get("extra") or {}
                color = extra.get("color_name", "desconocido")
                simple_dets.append(
                    f"{cls} (track_id={track_id}, color={color})"
                )
            frames_desc.append(
                f"Frame {rf['frame_number']}: " + "; ".join(simple_dets)
            )

        frames_text = "\n".join(frames_desc)

        system_prompt = """
Eres un analista de tráfico que responde preguntas sobre videos urbanos.
Tienes:
- Un resumen global del video (estadísticas de vehículos, colores, tráfico, posibles colisiones).
- Un conjunto de documentos relevantes generados via RAG (descripciones de vehículos, minutos del video y un resumen global).
- Una lista de frames representativos con sus detecciones (clase, track_id, color estimado).

Tu tarea:
- Combinar TODA la información disponible (resumen global + documentos RAG + descripciones de frames).
- Contestar la pregunta del usuario basándote solo en esa evidencia.
- Si la respuesta no está claramente soportada por la evidencia, indica "value": null y explica la incertidumbre.
- Idealmente responde en JSON con el formato:

{
  "answer": string,
  "type": "numeric" | "text" | "count" | "event" | "boolean",
  "value": number | string | boolean | null,
  "evidence": string[],
  "reasoning": string,
  "confidence": number
}
        """.strip()

        user_content = {
            "question": question,
            "global_summary": global_summary,
            "retrieved_docs": retrieved_docs or [],
            "frames_description": frames_text,
        }

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_content, ensure_ascii=False),
                },
            ],
            # si tu modelo no soporta esto, puedes quitar esta línea
            response_format={"type": "json_object"},
        )

        content = resp.choices[0].message.content or ""
        print("RAW LLM CONTENT:", content)

        # Intentar parsear como JSON
        try:
            parsed = json.loads(content)
            if "answer" not in parsed:
                parsed["answer"] = content
            return parsed
        except Exception:
            # Si no es JSON, devolver texto plano
            return {
                "answer": content,
                "type": "text",
                "value": content,
                "evidence": [],
                "reasoning": "El modelo devolvió una respuesta no JSON, se incluye el texto completo.",
                "confidence": 0.5,
            }

    except Exception as e:
        import traceback
        print("ERROR EN call_multimodal_video_llm:\n", traceback.format_exc())
        return {"error": str(e), "answer": None}


# ============================================================
#   ENDPOINT MULTIMODAL GLOBAL DEL VIDEO (CON RAG)
# ============================================================
@app.post("/multimodal/video-query")
async def multimodal_video_query(payload: VideoQuery = Body(...)):
    db: Session = SessionLocal()
    try:
        # 1) Construir o leer resumen global
        global_summary = build_global_video_summary(db, payload.video_id)

        # 2) Asegurar que el conocimiento del video esté indexado en la BD (RAG)
        docs_exist = db.query(VideoDocument).filter(
            VideoDocument.video_id == payload.video_id
        ).limit(1).first()

        if not docs_exist:
            index_video_knowledge(db, payload.video_id, global_summary)

        # 3) Recuperar documentos relevantes para la pregunta (RAG)
        retrieved_docs = rag_retrieve(db, payload.video_id, payload.question, k=8)

        # 4) Frames representativos
        frames = get_representative_frames(db, payload.video_id, payload.step)
        if not frames:
            raise ValueError("No se pudieron obtener frames representativos")

        # Usa solo unos pocos frames para no saturar al LLM
        frames_to_use = frames[:5]

        # 5) Llamada al LLM combinando RAG + resumen global + descripciones de frames
        llm_resp = call_multimodal_video_llm(
            representative_frames=frames_to_use,
            global_summary=global_summary,
            question=payload.question,
            retrieved_docs=retrieved_docs,
        )

        return {
            "video_id": str(payload.video_id),
            "question": payload.question,
            "used_frames": [f["frame_number"] for f in frames_to_use],
            "global_summary": global_summary,
            "retrieved_docs": retrieved_docs,
            "llm_response": llm_resp,
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        db.close()


# ============================================================
# === ENDPOINT: STREAM EN TIEMPO REAL CON GPU ===
# ============================================================
@app.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    file: Optional[str] = Query(None),
    video_id: Optional[UUID] = Query(None),
    send_frames: bool = True
):
    await websocket.accept()

    if not file:
        await websocket.send_text("ERROR: Falta parámetro 'file'")
        await websocket.close()
        return

    if video_id is None:
        video_id = 0

    db = SessionLocal()
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        await websocket.send_text("ERROR: No se puede abrir el video")
        await websocket.close()
        db.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = 1.0 / fps
    batch_buffer: List[Detection] = []
    BATCH_SIZE = 50

    async def insert_batch(batch):
        local_db = SessionLocal()
        try:
            local_db.bulk_save_objects(batch)
            local_db.commit()
        finally:
            local_db.close()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                device=DEVICE,
                half=torch.cuda.is_available(),
                imgsz=640,
                conf=0.35,
                iou=0.5,
            )
            res = results[0]
            detections_to_insert: List[Detection] = []
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if hasattr(res, "boxes") and len(res.boxes) > 0:
                for box in res.boxes:
                    xyxy = box.xyxy[0].tolist()
                    cls = int(box.cls)
                    conf = float(box.conf)
                    x1, y1, x2, y2 = map(int, xyxy)
                    track_id = int(box.id[0]) if hasattr(box, "id") and box.id is not None else None
                    class_name = model.names[cls] if hasattr(model, "names") else str(cls)

                    color_name, bgr_mean = vehicle_color_name(frame, x1, y1, x2, y2)

                    det = Detection(
                        video_id=video_id,
                        frame_number=frame_idx,
                        ts=datetime.fromtimestamp(time.time(), tz=timezone.utc),
                        object_class=f"{class_name} {conf:.2f}",
                        confidence=conf,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        track_id=track_id,
                        extra=json.dumps({
                            "color_name": color_name,
                            "bgr_mean": bgr_mean
                        }),
                    )
                    detections_to_insert.append(det)

                    if send_frames:
                        label = f"{class_name} {conf:.2f}" + (f" #{track_id}" if track_id else "")
                        label = f"{label} | {color_name}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            batch_buffer.extend(detections_to_insert)
            if len(batch_buffer) >= BATCH_SIZE:
                asyncio.create_task(insert_batch(batch_buffer.copy()))
                batch_buffer.clear()

            if send_frames:
                ok, buffer = cv2.imencode(".jpg", frame)
                if ok:
                    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
                    try:
                        await websocket.send_text(jpg_as_text)
                    except Exception:
                        break
            else:
                payload = [{
                    "frame": frame_idx,
                    "detections": [{
                        "cls": d.object_class,
                        "x1": d.x1, "y1": d.y1,
                        "x2": d.x2, "y2": d.y2,
                        "color": json.loads(d.extra)["color_name"]
                    } for d in detections_to_insert]
                }]
                try:
                    await websocket.send_text(json.dumps(payload))
                except Exception:
                    break

            elapsed = time.time() - t0
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)

        if batch_buffer:
            await insert_batch(batch_buffer)

        await websocket.send_text("<<EOF>>")
        await websocket.close()

    except Exception as e:
        try:
            await websocket.send_text(f"ERROR: {str(e)}")
            await websocket.close()
        except:
            pass
    finally:
        db.close()
        cap.release()
