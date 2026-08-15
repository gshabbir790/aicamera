
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from asset_manager import AssetManager
from landmark_service import LandmarkService
from skin_engine import SkinEngine
from suit_engine import SuitEngine


@dataclass
class PipelineOutput:
    image: np.ndarray
    face_detected: bool
    pose_detected: bool
    landmark_count: int
    landmark_confidence: float
    suit_applied: bool
    skin_retouch_applied: bool
    error_code: str | None


class PhotoPipeline:
    def __init__(
        self,
        landmarks: LandmarkService,
        assets: AssetManager,
    ):
        self.landmarks = landmarks
        self.assets = assets
        self.suit_engine = SuitEngine()
        self.skin_engine = SkinEngine()

    def process(
        self,
        image: np.ndarray,
        suit_id: str | None,
        smooth: float,
        brighten: float,
    ) -> PipelineOutput:
        lm = self.landmarks.detect(image)
        face_detected = bool(lm.face)
        pose_detected = bool(lm.pose)
        confidence = min(lm.face_visibility, lm.pose_visibility) if (
            face_detected and pose_detected
        ) else max(lm.face_visibility, lm.pose_visibility)

        if not face_detected:
            return PipelineOutput(
                image, False, pose_detected, 0, confidence,
                False, False, "NO_FACE",
            )

        if not pose_detected:
            return PipelineOutput(
                image, True, False, len(lm.face), confidence,
                False, False, "NO_POSE",
            )

        output = image.copy()
        suit_applied = False
        error_code = None

        if suit_id:
            try:
                metadata = self.assets.get(suit_id)
                rgba = self.assets.load_rgba(suit_id)
                suit_result = self.suit_engine.fit(
                    output, rgba, metadata, lm.pose, lm.face,
                    confidence_threshold=0.75,
                )
                output = suit_result.image
                suit_applied = suit_result.applied
                error_code = suit_result.error_code
            except (KeyError, OSError, ValueError) as exc:
                error_code = "INVALID_SUIT_ASSET"
                _ = exc

        output, skin_applied = self.skin_engine.apply(
            output, lm.face, smooth, brighten
        )

        return PipelineOutput(
            output,
            face_detected,
            pose_detected,
            len(lm.face) + len(lm.pose),
            confidence,
            suit_applied,
            skin_applied,
            error_code,
        )
