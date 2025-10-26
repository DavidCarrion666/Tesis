from sqlalchemy import Column, String, Float, ForeignKey, TIMESTAMP, Integer
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
import uuid

class Video(Base):
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    video_name = Column(String, index=True)
    source = Column(String, index=True)
    fps = Column(Integer)
    duration_s = Column(Float)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    detections = relationship("Detection", back_populates="video")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"))
    frame_number = Column(Integer, nullable=False)
    ts = Column(TIMESTAMP, nullable=False)
    object_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    track_id = Column(Integer, nullable=True)
    extra = Column(JSONB, default={})

    video = relationship("Video", back_populates="detections")
