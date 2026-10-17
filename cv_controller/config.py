"""Configuration settings for CV Sniper."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# ===================== FILE PATHS =====================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
FACE_MODEL = MODELS_DIR / "face_landmarker.task"

# ===================== CAMERA & DISPLAY =====================
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
CAMERA_BUFFER_SIZE: int = 1  # Minimizes latency / frame lag
WINDOW_TITLE: str = "CV Sniper - Optimized"

# ===================== GAMEPLAY & TARGETING =====================
TARGET_COUNT: int = 4
TARGET_RADIUS: Tuple[int, int] = (22, 34)
TARGET_SPEED: Tuple[float, float] = (75.0, 150.0)
TARGET_RESPAWN_MIN: float = 0.7
TARGET_RESPAWN_MAX: float = 1.5
TARGET_SCORE_VALUE: int = 100
HIT_TOLERANCE: int = 8

# ===================== SCOPE & CROSSHAIR =====================
SCOPE_RADIUS: int = 155
SCOPE_DARKNESS_ALPHA: float = 0.35
AIM_SMOOTHING: float = 0.30
FIRE_COOLDOWN: float = 0.80          # Minimum seconds between shots
SHOT_EFFECT_DURATION: float = 0.18   # Muzzle flash / recoil duration
HIT_MARKER_DURATION: float = 0.55    # Hit marker visibility duration
GESTURE_GRACE: float = 0.40          # Seconds to tolerate lost finger-gun before un-scoping
HAND_LOST_TIMEOUT: float = 1.0       # Seconds before returning to IDLE if hand is lost

# ===================== EYE & BLINK DETECTION =====================
CLOSED_EAR: float = 0.20
OPEN_EAR: float = 0.235
MIN_BLINK_DURATION: float = 0.05
MAX_BLINK_DURATION: float = 0.50
BLINK_COOLDOWN: float = 0.20

LEFT_EYE_INDICES: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)

# ===================== HAND CONNECTIONS =====================
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# ===================== MEDIAPIPE CONFIDENCE =====================
MIN_HAND_DETECTION_CONFIDENCE: float = 0.5
MIN_HAND_PRESENCE_CONFIDENCE: float = 0.5
MIN_HAND_TRACKING_CONFIDENCE: float = 0.5

MIN_FACE_DETECTION_CONFIDENCE: float = 0.5
MIN_FACE_PRESENCE_CONFIDENCE: float = 0.5
MIN_FACE_TRACKING_CONFIDENCE: float = 0.5

# ===================== COLOR PALETTE (BGR) =====================
COLOR_TEXT_PRIMARY = (255, 255, 255)
COLOR_TEXT_MUTED = (180, 190, 190)
COLOR_HUD_BG = (10, 18, 22)
COLOR_RETICLE_NORMAL = (50, 255, 90)
COLOR_RETICLE_RECOIL = (30, 40, 255)
COLOR_SCOPE_RING = (220, 235, 220)
COLOR_TARGET = (60, 220, 255)
COLOR_HIT = (60, 255, 90)
COLOR_MISS = (40, 80, 255)
COLOR_EQUIPPED = (0, 220, 255)
COLOR_IDLE = (180, 180, 180)
COLOR_SCOPED = (70, 255, 100)
COLOR_HAND_SKELETON = (70, 255, 100)
COLOR_HAND_JOINT = (40, 230, 255)
COLOR_EYE_OUTLINE = (255, 220, 60)
