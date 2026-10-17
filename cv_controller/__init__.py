"""CV Sniper controller and vision processing package."""

from . import config
from .detector import VisionDetector, VisionResult
from .game import GameEngine, GameState, GameStats, Target, TargetManager
from .gestures import BlinkDetector, detect_finger_gun, eye_aspect_ratio, hand_is_raised
from .renderer import OptimizedRenderer

__all__ = [
    "config",
    "VisionDetector",
    "VisionResult",
    "GameEngine",
    "GameState",
    "GameStats",
    "Target",
    "TargetManager",
    "BlinkDetector",
    "detect_finger_gun",
    "eye_aspect_ratio",
    "hand_is_raised",
    "OptimizedRenderer",
]
