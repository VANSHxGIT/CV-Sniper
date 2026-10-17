"""Gesture and eye tracking mathematical utilities for CV Sniper."""

import math
from typing import Any, List, Optional, Tuple, Union

from . import config


def distance(a: Any, b: Any) -> float:
    """Calculate 2D Euclidean distance between two landmarks or points."""
    ax = getattr(a, "x", a[0] if isinstance(a, (list, tuple)) else 0.0)
    ay = getattr(a, "y", a[1] if isinstance(a, (list, tuple)) else 0.0)
    bx = getattr(b, "x", b[0] if isinstance(b, (list, tuple)) else 0.0)
    by = getattr(b, "y", b[1] if isinstance(b, (list, tuple)) else 0.0)
    return math.hypot(ax - bx, ay - by)


def finger_extended(hand: List[Any], tip: int, pip: int, multiplier: float = 1.15) -> bool:
    """Determine whether a finger is extended relative to the wrist."""
    wrist = hand[0]
    return distance(hand[tip], wrist) > distance(hand[pip], wrist) * multiplier


def detect_finger_gun(hand: List[Any]) -> bool:
    """
    Detect finger gun gesture:
    - Index finger extended
    - Thumb extended upwards/outwards
    - Middle, ring, and pinky curled inward
    """
    index = finger_extended(hand, 8, 6)
    middle = finger_extended(hand, 12, 10)
    ring = finger_extended(hand, 16, 14)
    pinky = finger_extended(hand, 20, 18)
    thumb = distance(hand[4], hand[0]) > distance(hand[3], hand[0]) * 1.10

    return bool(index and thumb and not middle and not ring and not pinky)


def hand_is_raised(
    hand: List[Any],
    wrist_threshold: float = 0.75,
    palm_threshold: float = 0.65,
) -> bool:
    """Determine whether the player's hand is raised high enough in the frame."""
    wrist_y = hand[0].y
    palm_y = sum(hand[i].y for i in (0, 5, 9, 13, 17)) / 5.0
    return wrist_y < wrist_threshold or palm_y < palm_threshold


def eye_aspect_ratio(face: List[Any], indices: Tuple[int, ...]) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for a given eye landmark sequence.
    EAR drops significantly when an eye is closed.
    """
    p1, p2, p3, p4, p5, p6 = [face[i] for i in indices]
    horizontal = distance(p1, p4)

    if horizontal < 1e-6:
        return 1.0

    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * horizontal)


class BlinkDetector:
    """
    Robust stateful detector for intentional blinks (trigger pulls).
    Prevents multiple triggers per blink and enforces timing thresholds.
    """

    def __init__(
        self,
        closed_ear: float = config.CLOSED_EAR,
        open_ear: float = config.OPEN_EAR,
        min_duration: float = config.MIN_BLINK_DURATION,
        max_duration: float = config.MAX_BLINK_DURATION,
        cooldown: float = config.BLINK_COOLDOWN,
    ):
        self.closed_ear = closed_ear
        self.open_ear = open_ear
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.cooldown = cooldown

        self.closed_since: Optional[float] = None
        self.last_blink_time: float = -10.0
        self.state_string: str = "OPEN"

    def process(self, ear: Optional[float], now: float) -> Tuple[bool, str]:
        """
        Process current frame's average EAR and return (is_blink_event, eye_state_str).
        Blink event fires on eye reopening within valid duration window.
        """
        if ear is None:
            self.closed_since = None
            self.state_string = "UNKNOWN"
            return False, self.state_string

        blink_event = False

        if ear < self.closed_ear:
            self.state_string = "CLOSED"
            if self.closed_since is None:
                self.closed_since = now
        else:
            self.state_string = "OPEN"
            if self.closed_since is not None:
                duration = now - self.closed_since
                if self.min_duration <= duration <= self.max_duration:
                    if now - self.last_blink_time >= self.cooldown:
                        blink_event = True
                        self.last_blink_time = now
                self.closed_since = None

        return blink_event, self.state_string

    def reset(self) -> None:
        """Reset internal state."""
        self.closed_since = None
        self.last_blink_time = -10.0
        self.state_string = "OPEN"
