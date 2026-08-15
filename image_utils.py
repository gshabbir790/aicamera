
from __future__ import annotations

import base64
import binascii

import cv2
import numpy as np


def decode_image(data: bytes, max_pixels: int) -> np.ndarray:
    if not data:
        raise ValueError("EMPTY_IMAGE")
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("INVALID_IMAGE")
    h, w = image.shape[:2]
    if h * w > max_pixels:
        raise ValueError("IMAGE_TOO_LARGE")
    return image


def decode_data_url(value: str) -> bytes:
    if "," not in value:
        raise ValueError("INVALID_BASE64_IMAGE")
    header, payload = value.split(",", 1)
    if not header.startswith("data:image/"):
        raise ValueError("INVALID_DATA_URL")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("INVALID_BASE64_IMAGE") from exc


def encode_jpeg(image: np.ndarray, quality: int = 94) -> str:
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise ValueError("JPEG_ENCODING_FAILED")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def resize_max(image: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
