
from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from asset_manager import AssetManager
from config import settings
from image_utils import decode_data_url, decode_image, encode_jpeg, resize_max
from landmark_service import LandmarkService
from pipeline import PhotoPipeline
from schemas import (
    Base64ProcessRequest,
    Diagnostics,
    PreviewResponse,
    ProcessPhotoResponse,
    SuitCatalogItem,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("admission-photo-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.assets = AssetManager(settings.assets_root)
    app.state.landmarks = LandmarkService(
        settings.face_model_path,
        settings.pose_model_path,
    )
    app.state.pipeline = PhotoPipeline(
        app.state.landmarks,
        app.state.assets,
    )
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

origins = (
    ["*"]
    if settings.frontend_origin == "*"
    else [x.strip() for x in settings.frontend_origin.split(",") if x.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.get("/api/v1/health")
async def health(request: Request):
    return {
        "status": "ok",
        "service": settings.app_name,
        "runtime": "CPU",
        "gpu_available": False,
        "assets_loaded": len(request.app.state.assets.catalog()),
    }


@app.get("/api/v1/suits", response_model=list[SuitCatalogItem])
async def suits(request: Request):
    return request.app.state.assets.catalog()


async def _read_upload(upload: UploadFile) -> bytes:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(415, "IMAGE_CONTENT_TYPE_REQUIRED")
    data = await upload.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "UPLOAD_TOO_LARGE")
    return data


async def _run_pipeline(
    request: Request,
    image,
    suit_id: str | None,
    smooth: float,
    brighten: float,
    preview: bool,
):
    started = time.perf_counter()
    if preview:
        image = resize_max(image, settings.preview_max_dimension)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                request.app.state.pipeline.process,
                image,
                suit_id,
                smooth,
                brighten,
            ),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "PROCESSING_TIMEOUT") from exc

    elapsed = int((time.perf_counter() - started) * 1000)
    metadata = Diagnostics(
        face_detected=result.face_detected,
        pose_detected=result.pose_detected,
        landmark_count=result.landmark_count,
        landmark_confidence=round(result.landmark_confidence, 4),
        suit_applied=result.suit_applied,
        skin_retouch_applied=result.skin_retouch_applied,
        processing_time_ms=elapsed,
        error_code=result.error_code,
    )

    # Preserve original image if fitting/detection failed.
    encoded = encode_jpeg(result.image, 75 if preview else 94)
    return encoded, metadata


@app.post("/api/v1/preview-suit", response_model=PreviewResponse)
async def preview_suit(
    request: Request,
    image: Annotated[UploadFile, File(...)],
    suit_id: Annotated[str | None, Form()] = None,
    skin_smooth_level: Annotated[float, Form(ge=0.0, le=1.0)] = 0.0,
    brighten_level: Annotated[float, Form(ge=0.0, le=1.0)] = 0.0,
):
    data = await _read_upload(image)
    decoded = decode_image(data, settings.max_image_pixels)
    encoded, metadata = await _run_pipeline(
        request, decoded, suit_id, skin_smooth_level,
        brighten_level, preview=True,
    )
    return PreviewResponse(
        success=metadata.error_code not in {
            "NO_FACE", "NO_POSE", "INVALID_SUIT_ASSET"
        },
        processed_image=encoded,
        metadata=metadata,
        error_message=metadata.error_code,
    )


@app.post("/api/v1/process-photo", response_model=ProcessPhotoResponse)
async def process_photo(
    request: Request,
    image: Annotated[UploadFile, File(...)],
    suit_id: Annotated[str | None, Form()] = None,
    skin_smooth_level: Annotated[float, Form(ge=0.0, le=1.0)] = 0.3,
    brighten_level: Annotated[float, Form(ge=0.0, le=1.0)] = 0.1,
    background_mode: Annotated[str, Form()] = "client_background",
):
    if background_mode not in {"client_background", "server_background"}:
        raise HTTPException(422, "INVALID_BACKGROUND_MODE")
    data = await _read_upload(image)
    decoded = decode_image(data, settings.max_image_pixels)
    encoded, metadata = await _run_pipeline(
        request, decoded, suit_id, skin_smooth_level,
        brighten_level, preview=False,
    )
    return ProcessPhotoResponse(
        success=metadata.error_code not in {
            "NO_FACE", "NO_POSE", "INVALID_SUIT_ASSET"
        },
        processed_image=encoded,
        metadata=metadata,
        error_message=metadata.error_code,
    )


@app.post("/api/v1/process-photo/base64", response_model=ProcessPhotoResponse)
async def process_photo_base64(
    request: Request,
    payload: Base64ProcessRequest,
):
    data = decode_data_url(payload.image)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "UPLOAD_TOO_LARGE")
    decoded = decode_image(data, settings.max_image_pixels)
    encoded, metadata = await _run_pipeline(
        request,
        decoded,
        payload.suit_id,
        payload.skin_smooth_level,
        payload.brighten_level,
        preview=False,
    )
    return ProcessPhotoResponse(
        success=metadata.error_code not in {
            "NO_FACE", "NO_POSE", "INVALID_SUIT_ASSET"
        },
        processed_image=encoded,
        metadata=metadata,
        error_message=metadata.error_code,
    )
