
from __future__ import annotations

import cv2
import numpy as np


class SkinEngine:
    """Selective, edge-preserving facial retouching."""

    # Canonical MediaPipe Face Mesh regions. These are protective landmarks,
    # not a claim that the mesh is a semantic skin classifier.
    PROTECTED = {
        "left_eye": list(range(33, 133)),
        "right_eye": list(range(263, 362)),
        "mouth": list(range(61, 292)),
        "brows": list(range(70, 300)),
    }

    def apply(
        self,
        image: np.ndarray,
        face_landmarks: list[tuple[float, float, float]],
        smooth_level: float,
        brighten_level: float,
    ) -> tuple[np.ndarray, bool]:
        if not face_landmarks or smooth_level <= 0 and brighten_level <= 0:
            return image, False

        h, w = image.shape[:2]
        mask = self._skin_mask(face_landmarks, w, h)
        if cv2.countNonZero(mask) < 100:
            return image, False

        strength = float(np.clip(smooth_level, 0.0, 1.0))
        diameter = int(5 + 8 * strength)
        sigma = 20 + 55 * strength

        smoothed = cv2.bilateralFilter(
            image, diameter, sigma, sigma
        )

        # High-pass texture extraction.
        low = cv2.GaussianBlur(image, (0, 0), 2.0)
        texture = cv2.addWeighted(image, 1.35, low, -0.35, 0)

        # Retain 15–25% facial micro-texture depending on smoothing.
        texture_weight = 0.25 - 0.10 * strength
        textured = cv2.addWeighted(
            smoothed, 1.0 - texture_weight, texture, texture_weight, 0
        )

        out = image.astype(np.float32)
        target = textured.astype(np.float32)

        if brighten_level > 0:
            hsv = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2HSV)
            value = hsv[:, :, 2].astype(np.float32)
            value += 18.0 * float(np.clip(brighten_level, 0.0, 1.0))
            hsv[:, :, 2] = np.clip(value, 0, 255).astype(np.uint8)
            target = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

        alpha = (mask.astype(np.float32) / 255.0) * max(
            strength, float(np.clip(brighten_level, 0.0, 1.0))
        )
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)[..., None]
        out = out * (1.0 - alpha) + target * alpha
        return np.clip(out, 0, 255).astype(np.uint8), True

    def _skin_mask(
        self,
        landmarks: list[tuple[float, float, float]],
        w: int,
        h: int,
    ) -> np.ndarray:
        def p(idx: int) -> tuple[int, int]:
            x, y, _ = landmarks[idx]
            return int(x * w), int(y * h)

        if len(landmarks) < 455:
            return np.zeros((h, w), dtype=np.uint8)

        # Broad face oval from canonical outer face landmarks.
        outer = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                 162, 21, 54, 103, 67, 109]
        poly = np.array([p(i) for i in outer], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        # Exclude eyes, eyebrows, nostrils and mouth with generous polygons.
        exclusions = [
            [33, 133, 160, 159, 158, 157, 173, 155, 154, 153],
            [263, 362, 387, 386, 385, 384, 398, 382, 381, 380],
            [70, 63, 105, 66, 107, 55, 65, 52],
            [336, 296, 334, 293, 300, 285, 295, 282],
            [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415],
            [1, 2, 98, 327],
        ]
        for region in exclusions:
            pts = np.array([p(i) for i in region if i < len(landmarks)])
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 0)

        return mask
