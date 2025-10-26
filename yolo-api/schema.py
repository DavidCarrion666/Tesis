from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class DetectionBase(BaseModel):
    frame_number: int
    ts: datetime
    object_class: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: Optional[int] = None
    extra: Optional[dict] = {}

class DetectionCreate(DetectionBase):
    video_id: int

class VideoBase(BaseModel):
    video_name: str
    source: str
    fps: int
    duration_s: float

class VideoCreate(VideoBase):
    pass

class Video(VideoBase):
    id: int
    created_at: datetime
    detections: List[DetectionBase] = []

    class Config:
        orm_mode = True
