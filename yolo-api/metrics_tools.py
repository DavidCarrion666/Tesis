# metrics_tools.py
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict
from uuid import UUID as UUIDcls

from sqlalchemy.orm import Session

from models import Video, Detection
import math


# ------------------ Helpers ------------------

def _to_uuid(s: str):
    try:
        return UUIDcls(s)
    except Exception:
        # Si no es UUID válido, devuelve tal cual (por si la columna no es UUID en DB)
        return s

def _normalize_class(object_class: str) -> str:
    """
    Tu schema guarda en Detection.object_class algo como: 'car 0.87'.
    Nos quedamos con la primera 'palabra' como clase.
    """
    if not object_class:
        return "unknown"
    return object_class.split()[0].strip().lower()

def _centroid(det: Detection) -> Tuple[float, float]:
    cx = (float(det.x1) + float(det.x2)) / 2.0
    cy = (float(det.y1) + float(det.y2)) / 2.0
    return cx, cy


# ------------------ API ------------------

def get_summary(db: Session, video_id: str) -> Dict[str, Any]:
    vid = _to_uuid(video_id)

    video = db.query(Video).filter(Video.id == vid).first()
    if not video:
        return {"error": f"video_id no encontrado: {video_id}"}

    q = db.query(Detection).filter(Detection.video_id == vid)
    dets: List[Detection] = q.all()

    total = len(dets)
    if total == 0:
        return {
            "video_id": video_id,
            "total_detections": 0,
            "unique_tracks": 0,
            "classes": {},
            "estimated_duration_s": video.duration_s,  # lo que tengas en la tabla
            "frames_max": None,
            "fps": video.fps,
        }

    # frames / duración
    max_frame = max(d.frame_number for d in dets if d.frame_number is not None)
    fps = video.fps or 0
    est_duration = (max_frame / fps) if (fps and max_frame is not None) else video.duration_s

    # tracks únicos (ignora Nones)
    track_ids = {d.track_id for d in dets if d.track_id is not None}
    unique_tracks = len(track_ids)

    # conteo por clase normalizada
    class_counts: Counter = Counter(_normalize_class(d.object_class) for d in dets)

    return {
        "video_id": video_id,
        "total_detections": total,
        "unique_tracks": unique_tracks,
        "classes": dict(class_counts),
        "estimated_duration_s": est_duration,
        "frames_max": max_frame,
        "fps": fps,
    }


def get_colors(db: Session, video_id: str) -> Dict[str, Any]:
    vid = _to_uuid(video_id)
    q = db.query(Detection).filter(Detection.video_id == vid)
    dets: List[Detection] = q.all()

    if not dets:
        return {"video_id": video_id, "colors_by_class": {}}

    # contar color por clase normalizada
    agg: Dict[str, Counter] = defaultdict(Counter)

    for d in dets:
        klass = _normalize_class(d.object_class)
        color = None
        if d.extra and isinstance(d.extra, dict):
            color = d.extra.get("color_name")
        if not color:
            color = "desconocido"
        agg[klass][color] += 1

    # ordenar por frecuencia
    out: Dict[str, List[Dict[str, Any]]] = {}
    for klass, cnt in agg.items():
        out[klass] = [{"color": c, "count": n} for c, n in cnt.most_common()]

    return {"video_id": video_id, "colors_by_class": out}


def get_tracks(db: Session, video_id: str, klass: Optional[str] = None) -> Dict[str, Any]:
    vid = _to_uuid(video_id)
    q = db.query(Detection).filter(Detection.video_id == vid, Detection.track_id.isnot(None))
    if klass:
        klass = klass.strip().lower()
        # filtrar por clase normalizada
        dets = [d for d in q.all() if _normalize_class(d.object_class) == klass]
    else:
        dets = q.all()

    if not dets:
        return {"video_id": video_id, "tracks": []}

    # agrupar por track_id
    by_track: Dict[int, List[Detection]] = defaultdict(list)
    for d in dets:
        by_track[int(d.track_id)].append(d)

    tracks_out = []
    for tid, items in by_track.items():
        frames = [d.frame_number for d in items if d.frame_number is not None]
        if not frames:
            duration_frames = None
            first_frame = None
            last_frame = None
        else:
            first_frame = min(frames)
            last_frame = max(frames)
            duration_frames = (last_frame - first_frame + 1)

        # clase predominante para el track
        class_counts = Counter(_normalize_class(d.object_class) for d in items)
        main_class, _ = class_counts.most_common(1)[0]

        tracks_out.append({
            "track_id": tid,
            "class": main_class,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "duration_frames": duration_frames,
            "count_detections": len(items),
        })

    # ordenar por track_id
    tracks_out.sort(key=lambda x: x["track_id"])
    return {"video_id": video_id, "tracks": tracks_out}


def get_trajectory(db: Session, video_id: str, track_id: int) -> Dict[str, Any]:
    vid = _to_uuid(video_id)
    q = (db.query(Detection)
           .filter(Detection.video_id == vid, Detection.track_id == track_id)
           .order_by(Detection.frame_number.asc()))
    dets: List[Detection] = q.all()

    if not dets:
        return {"video_id": video_id, "track_id": track_id, "trajectory": []}

    traj = []
    for d in dets:
        cx, cy = _centroid(d)
        traj.append({
            "frame": d.frame_number,
            "cx": float(cx),
            "cy": float(cy),
        })

    return {"video_id": video_id, "track_id": track_id, "trajectory": traj}


def get_speed(db: Session, video_id: str, track_id: int) -> Dict[str, Any]:
    vid = _to_uuid(video_id)

    video = db.query(Video).filter(Video.id == vid).first()
    if not video:
        return {"error": f"video_id no encontrado: {video_id}"}

    fps = video.fps or 0
    if not fps:
        return {"video_id": video_id, "track_id": track_id, "error": "No hay FPS en el video para calcular velocidad."}

    q = (db.query(Detection)
           .filter(Detection.video_id == vid, Detection.track_id == track_id)
           .order_by(Detection.frame_number.asc()))
    dets: List[Detection] = q.all()

    if len(dets) < 2:
        return {"video_id": video_id, "track_id": track_id, "error": "Se requieren al menos 2 puntos para calcular velocidad."}

    # distancias por paso (entre frames consecutivos del mismo track)
    step_dists: List[float] = []
    prev_cx, prev_cy = _centroid(dets[0])
    prev_frame = dets[0].frame_number

    for d in dets[1:]:
        cx, cy = _centroid(d)
        df = max(1, (d.frame_number - prev_frame) if (prev_frame is not None and d.frame_number is not None) else 1)
        # distancia euclídea entre centroides
        dist = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
        # normalizamos por el salto de frames (por si hay gaps)
        step_dists.append(dist / df)
        prev_cx, prev_cy = cx, cy
        prev_frame = d.frame_number

    if not step_dists:
        return {"video_id": video_id, "track_id": track_id, "error": "Sin pasos válidos para velocidad."}

    avg_step = sum(step_dists) / len(step_dists)      # px por frame
    max_step = max(step_dists)                         # px por frame

    speed_avg_px_s = avg_step * fps
    speed_peak_px_s = max_step * fps

    return {
        "video_id": video_id,
        "track_id": track_id,
        "fps": fps,
        "samples": len(step_dists),
        "speed_avg_px_s": float(speed_avg_px_s),
        "speed_peak_px_s": float(speed_peak_px_s),
    }
