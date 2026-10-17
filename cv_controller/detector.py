"""MediaPipe vision detection engine for hand and face tracking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from . import config
from .gestures import detect_finger_gun, eye_aspect_ratio, hand_is_raised


@dataclass
class VisionResult:
    """Consolidated vision analysis for a single video frame."""
    hand_seen: bool
    face_seen: bool
    hand: Optional[List[Any]]
    face: Optional[List[Any]]
    hand_raised: bool
    finger_gun: bool
    ear: float
    aim_point: Optional[Tuple[float, float]]  # (x, y) in screen pixels


class VisionDetector:
    """Manages MediaPipe task models, frame conversions, and gesture evaluations."""

    def __init__(
        self,
        hand_model_path: Path = config.HAND_MODEL,
        face_model_path: Path = config.FACE_MODEL,
    ):
        self.hand_model_path = Path(hand_model_path)
        self.face_model_path = Path(face_model_path)

        self._validate_model_paths()

        self.hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.hand_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_HAND_TRACKING_CONFIDENCE,
        )

        self.face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.face_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=config.MIN_FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=config.MIN_FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_FACE_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.hand_detector = vision.HandLandmarker.create_from_options(self.hand_options)
        self.face_detector = vision.FaceLandmarker.create_from_options(self.face_options)

        self._rgb_buffer: Optional[np.ndarray] = None

    def _validate_model_paths(self) -> None:
        """Verify that MediaPipe task model files exist."""
        if not self.hand_model_path.exists():
            raise FileNotFoundError(
                f"Hand model file not found at: {self.hand_model_path}.\n"
                "Please ensure hand_landmarker.task is located in the models/ directory."
            )
        if not self.face_model_path.exists():
            raise FileNotFoundError(
                f"Face model file not found at: {self.face_model_path}.\n"
                "Please ensure face_landmarker.task is located in the models/ directory."
            )

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> VisionResult:
        """
        Run inference on video frame and extract hand and facial landmarks.
        Uses preallocated buffer to avoid memory thrashing.
        """
        h, w = frame_bgr.shape[:2]

        # Allocate or reuse contiguous RGB buffer
        if self._rgb_buffer is None or self._rgb_buffer.shape != frame_bgr.shape:
            self._rgb_buffer = np.empty_like(frame_bgr)

        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB, dst=self._rgb_buffer)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self._rgb_buffer)

        hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
        face_result = self.face_detector.detect_for_video(mp_image, timestamp_ms)

        hand_seen = bool(hand_result.hand_landmarks)
        face_seen = bool(face_result.face_landmarks)
        hand = hand_result.hand_landmarks[0] if hand_seen else None
        face = face_result.face_landmarks[0] if face_seen else None

        raised = hand_is_raised(hand) if hand_seen else False
        finger_gun = detect_finger_gun(hand) if hand_seen else False

        aim_point: Optional[Tuple[float, float]] = None
        if hand_seen and hand is not None:
            raw_x = min(max(hand[8].x, 0.0), 1.0) * w
            raw_y = min(max(hand[8].y, 0.0), 1.0) * h
            aim_point = (raw_x, raw_y)

        ear = 1.0
        if face_seen and face is not None:
            ear = (
                eye_aspect_ratio(face, config.LEFT_EYE_INDICES)
                + eye_aspect_ratio(face, config.RIGHT_EYE_INDICES)
            ) / 2.0

        return VisionResult(
            hand_seen=hand_seen,
            face_seen=face_seen,
            hand=hand,
            face=face,
            hand_raised=raised,
            finger_gun=finger_gun,
            ear=ear,
            aim_point=aim_point,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "hand_detector") and self.hand_detector:
            self.hand_detector.close()
        if hasattr(self, "face_detector") and self.face_detector:
            self.face_detector.close()

    def __enter__(self) -> "VisionDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
