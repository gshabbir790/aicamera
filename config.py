
from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Admission Photo AI Service")
    api_prefix: str = "/api/v1"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _int_env("PORT", 8000)
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "*")
    request_timeout_seconds: float = _float_env("REQUEST_TIMEOUT_SECONDS", 45.0)
    max_upload_bytes: int = _int_env("MAX_UPLOAD_BYTES", 15 * 1024 * 1024)
    max_image_pixels: int = _int_env("MAX_IMAGE_PIXELS", 25_000_000)
    preview_max_dimension: int = _int_env("PREVIEW_MAX_DIMENSION", 720)
    face_model_path: str = os.getenv(
        "FACE_MODEL_PATH", "models/face_landmarker.task"
    )
    pose_model_path: str = os.getenv(
        "POSE_MODEL_PATH", "models/pose_landmarker_full.task"
    )
    assets_root: str = os.getenv("ASSETS_ROOT", "assets/suits")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
