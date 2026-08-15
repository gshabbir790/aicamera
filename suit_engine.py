
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class SuitResult:
    image: np.ndarray
    applied: bool
    confidence: float
    method: str
    error_code: str | None = None


def _pt(landmarks: list[tuple[float, float, float]], idx: int, w: int, h: int):
    if idx < 0 or idx >= len(landmarks):
        raise IndexError(idx)
    x, y, _ = landmarks[idx]
    return np.array([x * w, y * h], dtype=np.float32)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _alpha_blend(
    background: np.ndarray,
    overlay: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    alpha3 = np.repeat(alpha[..., None], 3, axis=2).astype(np.float32)
    out = (
        overlay[:, :, :3].astype(np.float32) * alpha3
        + background.astype(np.float32) * (1.0 - alpha3)
    )
    return np.clip(out, 0, 255).astype(np.uint8)


class SuitEngine:
    """Landmark-driven RGBA suit compositor."""

    # MediaPipe Pose indices:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8

    def fit(
        self,
        image: np.ndarray,
        suit_rgba: np.ndarray,
        metadata: dict[str, Any],
        pose: list[tuple[float, float, float]],
        face: list[tuple[float, float, float]],
        confidence_threshold: float = 0.75,
    ) -> SuitResult:
        if len(pose) <= self.RIGHT_SHOULDER:
            return SuitResult(
                image, False, 0.0, "none", "INSUFFICIENT_POSE_LANDMARKS"
            )

        h, w = image.shape[:2]
        try:
            left_shoulder = _pt(pose, self.LEFT_SHOULDER, w, h)
            right_shoulder = _pt(pose, self.RIGHT_SHOULDER, w, h)
            nose = _pt(pose, self.NOSE, w, h)
        except IndexError:
            return SuitResult(image, False, 0.0, "none", "LANDMARK_INDEX_ERROR")

        shoulder_width = _distance(left_shoulder, right_shoulder)
        if shoulder_width < max(20.0, w * 0.05):
            return SuitResult(image, False, 0.0, "none", "SHOULDER_GEOMETRY_INVALID")

        shoulder_angle = math.atan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0],
        )
        confidence = 0.80
        method = "perspective"

        # Face landmarks provide a neck/ear reference. FaceMesh landmark 152
        # is chin and 234/454 are lateral face references in the canonical mesh.
        if len(face) > 454:
            chin = _pt(face, 152, w, h)
            face_left = _pt(face, 234, w, h)
            face_right = _pt(face, 454, w, h)
            neck_center = (chin + (face_left + face_right) * 0.5) * 0.5
        else:
            # Explicitly use pose nose only as a vertical alignment reference.
            shoulder_mid = (left_shoulder + right_shoulder) * 0.5
            neck_center = shoulder_mid * 0.7 + nose * 0.3
            confidence = min(confidence, 0.70)

        src = np.float32([
            metadata["shoulder_left_anchor"],
            metadata["shoulder_right_anchor"],
            metadata["neck_center_anchor"],
            metadata["torso_bottom_anchor"],
        ])

        torso_bottom = shoulder_mid = (left_shoulder + right_shoulder) * 0.5
        torso_bottom = shoulder_mid + np.array(
            [0.0, shoulder_width * 1.25], dtype=np.float32
        )

        # Destination quadrilateral. The lower anchors are inferred from
        # shoulder geometry; this keeps the engine deterministic without
        # inventing unavailable anatomical landmarks.
        shoulder_vec = right_shoulder - left_shoulder
        perp = np.array([-shoulder_vec[1], shoulder_vec[0]], dtype=np.float32)
        perp_norm = np.linalg.norm(perp) or 1.0
        perp /= perp_norm

        lower_left = left_shoulder + np.array(
            [0.0, shoulder_width * 1.30], dtype=np.float32
        )
        lower_right = right_shoulder + np.array(
            [0.0, shoulder_width * 1.30], dtype=np.float32
        )

        dst = np.float32([
            left_shoulder,
            right_shoulder,
            neck_center,
            (lower_left + lower_right) * 0.5,
        ])

        # Rotate the template around its center to match shoulder roll.
        rotated = self._rotate_rgba(suit_rgba, -math.degrees(shoulder_angle))
        src_w = rotated.shape[1]
        src_h = rotated.shape[0]

        # If the supplied anchors are normalized/template coordinates,
        # metadata values <= 1 are interpreted as normalized.
        src_scaled = src.copy()
        if np.max(np.abs(src_scaled)) <= 1.5:
            src_scaled[:, 0] *= src_w
            src_scaled[:, 1] *= src_h

        if confidence >= confidence_threshold:
            try:
                matrix = cv2.getPerspectiveTransform(src_scaled, dst)
                warped = cv2.warpPerspective(
                    rotated,
                    matrix,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0),
                )
            except cv2.error:
                return self._affine_fallback(
                    image, rotated, src_scaled, dst, confidence
                )
        else:
            return self._affine_fallback(
                image, rotated, src_scaled[:3], dst[:3], confidence
            )

        result = self._composite(image, warped)
        return SuitResult(result, True, confidence, method)

    @staticmethod
    def _affine_fallback(
        image: np.ndarray,
        rgba: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
        confidence: float,
    ) -> SuitResult:
        try:
            matrix = cv2.getAffineTransform(
                src.astype(np.float32),
                dst.astype(np.float32),
            )
            h, w = image.shape[:2]
            warped = cv2.warpAffine(
                rgba,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            return SuitResult(
                SuitEngine._composite(image, warped),
                True,
                confidence,
                "affine",
            )
        except cv2.error:
            return SuitResult(
                image, False, confidence, "none", "AFFINE_TRANSFORM_FAILED"
            )

    @staticmethod
    def _rotate_rgba(image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    @staticmethod
    def _composite(background: np.ndarray, rgba: np.ndarray) -> np.ndarray:
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)
        return _alpha_blend(background, rgba, alpha)
