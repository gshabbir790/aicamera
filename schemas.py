
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Base64ProcessRequest(BaseModel):
    image: str = Field(min_length=32)
    suit_id: str | None = None
    skin_smooth_level: float = Field(default=0.3, ge=0.0, le=1.0)
    brighten_level: float = Field(default=0.1, ge=0.0, le=1.0)
    background_mode: Literal["client_background", "server_background"] = (
        "client_background"
    )


class Diagnostics(BaseModel):
    face_detected: bool
    pose_detected: bool
    landmark_count: int = 0
    landmark_confidence: float = 0.0
    suit_applied: bool = False
    skin_retouch_applied: bool = False
    processing_time_ms: int = 0
    error_code: str | None = None


class ProcessPhotoResponse(BaseModel):
    success: bool
    processed_image: str
    metadata: Diagnostics
    error_message: str | None = None


class PreviewResponse(BaseModel):
    success: bool
    processed_image: str
    metadata: Diagnostics
    error_message: str | None = None


class SuitCatalogItem(BaseModel):
    id: str
    category: str
    filename: str
    thumbnail: str | None = None
