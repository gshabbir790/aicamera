
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

LOGGER = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision


@dataclass
class LandmarkResult:
    face: list[tuple[float, float, float]]
    face_visibility: float
    pose: list[tuple[float, float, float]]
    pose_visibility: float


class LandmarkService:
    """Lazy-loaded MediaPipe Tasks API FaceLandmarker + PoseLandmarker."""

    def __init__(self, face_model_path: str, pose_model_path: str):
        self.face_model_path = Path(face_model_path)
        self.pose_model_path = Path(pose_model_path)
        self._face = None
        self._pose = None

    def _load(self) -> None:
        if self._face is None:
            if not self.face_model_path.exists():
                raise FileNotFoundError(
                    f"Face landmark model missing: {self.face_model_path}"
                )
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(self.face_model_path)
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._face = vision.FaceLandmarker.create_from_options(options)

        if self._pose is None:
            if not self.pose_model_path.exists():
                raise FileNotFoundError(
                    f"Pose landmark model missing: {self.pose_model_path}"
                )
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(self.pose_model_path)
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._pose = vision.PoseLandmarker.create_from_options(options)

    def detect(self, bgr: np.ndarray) -> LandmarkResult:
        self._load()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = self._face.detect(mp_image)
        pose_result = self._pose.detect(mp_image)

        face = []
        face_conf = 0.0
        if face_result.face_landmarks:
            face = [
                (float(p.x), float(p.y), float(p.z))
                for p in face_result.face_landmarks[0]
            ]
            face_conf = 1.0

        pose = []
        pose_conf = 0.0
        if pose_result.pose_landmarks:
            pose = [
                (float(p.x), float(p.y), float(p.z))
                for p in pose_result.pose_landmarks[0]
            ]
            pose_conf = self._pose_confidence(pose_result)

        return LandmarkResult(face, face_conf, pose, pose_conf)

    @staticmethod
    def _pose_confidence(result) -> float:
        if not result.pose_landmarks:
            return 0.0
        visibility = [
            float(getattr(p, "visibility", 1.0))
            for p in result.pose_landmarks[0]
        ]
        return sum(visibility) / max(1, len(visibility))
