"""Game logic, targeting engine, and state machine for CV Sniper."""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import random
from typing import List, Optional, Tuple

from . import config


class GameState(Enum):
    """Player weapon and targeting states."""
    IDLE = auto()
    EQUIPPED = auto()
    SCOPED = auto()


@dataclass
class Target:
    """Individual dynamic target."""
    x: float
    y: float
    r: int
    vx: float
    vy: float
    alive: bool = True
    respawn: float = 0.0

    @classmethod
    def create_random(cls, width: int, height: int) -> "Target":
        """Generate a new target with randomized size, position, and velocity."""
        r = random.randint(*config.TARGET_RADIUS)
        margin = r + 12
        x = float(random.randint(margin, max(margin, width - margin - 1)))
        y_min = min(height - 1, margin + 60)
        y_max = max(y_min, height - margin - 15)
        y = float(random.randint(y_min, y_max))

        angle = random.uniform(0, math.tau)
        speed = random.uniform(*config.TARGET_SPEED)

        return cls(
            x=x,
            y=y,
            r=r,
            vx=math.cos(angle) * speed,
            vy=math.sin(angle) * speed,
            alive=True,
            respawn=0.0,
        )

    def update(self, width: int, height: int, dt: float, now: float) -> None:
        """Update target position, physics, boundaries, and respawn timer."""
        if not self.alive:
            if now >= self.respawn:
                new_t = Target.create_random(width, height)
                self.x = new_t.x
                self.y = new_t.y
                self.r = new_t.r
                self.vx = new_t.vx
                self.vy = new_t.vy
                self.alive = True
                self.respawn = 0.0
            return

        self.x += self.vx * dt
        self.y += self.vy * dt
        r = float(self.r)

        # X boundary collision bounce
        if self.x < r:
            self.x = r
            self.vx = abs(self.vx)
        elif self.x > width - r:
            self.x = float(width - r)
            self.vx = -abs(self.vx)

        # Y boundary collision bounce (keeping targets in playable vertical band)
        low_y = min(float(height - r), r + 45.0)
        high_y = max(low_y, float(height - r - 10))

        if self.y < low_y:
            self.y = low_y
            self.vy = abs(self.vy)
        elif self.y > high_y:
            self.y = high_y
            self.vy = -abs(self.vy)

    def is_hit(self, shot_x: float, shot_y: float, tolerance: float = config.HIT_TOLERANCE) -> bool:
        """Check whether a shot center hits this target."""
        if not self.alive:
            return False
        dist = math.hypot(shot_x - self.x, shot_y - self.y)
        return dist <= (self.r + tolerance)


class TargetManager:
    """Manages active targets, updates physics, and resolves hits."""

    def __init__(self, count: int = config.TARGET_COUNT):
        self.count = count
        self.targets: List[Target] = []

    def ensure_initialized(self, width: int, height: int) -> None:
        """Populate initial target list if empty."""
        if not self.targets:
            self.targets = [Target.create_random(width, height) for _ in range(self.count)]

    def update(self, width: int, height: int, dt: float, now: float) -> None:
        """Update all managed targets."""
        self.ensure_initialized(width, height)
        for target in self.targets:
            target.update(width, height, dt, now)

    def check_hit(self, shot_x: float, shot_y: float) -> Optional[int]:
        """Check if shot hit any target. Returns index of hit target, or None."""
        for i, target in enumerate(self.targets):
            if target.is_hit(shot_x, shot_y):
                return i
        return None

    def reset(self, width: int, height: int) -> None:
        """Reset all targets."""
        self.targets = [Target.create_random(width, height) for _ in range(self.count)]


@dataclass
class GameStats:
    """Tracks player score, shots, and accuracy."""
    score: int = 0
    shots: int = 0
    hits: int = 0

    @property
    def accuracy(self) -> float:
        """Hit accuracy as percentage."""
        return (self.hits / self.shots * 100.0) if self.shots > 0 else 0.0

    def record_shot(self, is_hit: bool, score_delta: int = config.TARGET_SCORE_VALUE) -> None:
        """Record a fired shot."""
        self.shots += 1
        if is_hit:
            self.hits += 1
            self.score += score_delta

    def reset(self) -> None:
        """Reset stats to zero."""
        self.score = 0
        self.shots = 0
        self.hits = 0


class GameEngine:
    """Central game engine governing state transitions, aiming, and firing."""

    def __init__(self):
        self.state: GameState = GameState.IDLE
        self.stats = GameStats()
        self.target_manager = TargetManager()

        self.aim_x: Optional[float] = None
        self.aim_y: Optional[float] = None

        self.last_shot_time: Optional[float] = None
        self.shot_effect_time: float = -10.0

        self.last_hand_seen: float = -10.0
        self.last_gesture_seen: float = -10.0

        self.last_result_time: float = -10.0
        self.last_result_center: Tuple[int, int] = (0, 0)
        self.last_result_label: str = ""
        self.last_result_color: Tuple[int, int, int] = config.COLOR_HIT

        self.action_message: str = "READY"

    def update_aim(self, raw_x: float, raw_y: float) -> None:
        """Smoothly update aim crosshair position."""
        if self.aim_x is None or self.aim_y is None:
            self.aim_x, self.aim_y = raw_x, raw_y
        else:
            self.aim_x += (raw_x - self.aim_x) * config.AIM_SMOOTHING
            self.aim_y += (raw_y - self.aim_y) * config.AIM_SMOOTHING

    def update_states(
        self,
        hand_seen: bool,
        hand_raised: bool,
        finger_gun: bool,
        now: float,
    ) -> None:
        """Update game state machine based on gestures with timing grace periods."""
        if hand_seen:
            self.last_hand_seen = now
        if finger_gun:
            self.last_gesture_seen = now

        if self.state == GameState.IDLE:
            if hand_seen and finger_gun:
                self.state = GameState.SCOPED
                self.action_message = "SCOPE ACQUIRED"
            elif hand_seen and hand_raised:
                self.state = GameState.EQUIPPED
                self.action_message = "RIFLE EQUIPPED"

        elif self.state == GameState.EQUIPPED:
            if finger_gun:
                self.state = GameState.SCOPED
                self.action_message = "SCOPE ACQUIRED"
            elif now - self.last_hand_seen > config.HAND_LOST_TIMEOUT:
                self.state = GameState.IDLE
                self.action_message = "HAND LOST"

        elif self.state == GameState.SCOPED:
            if now - self.last_hand_seen > config.HAND_LOST_TIMEOUT:
                self.state = GameState.IDLE
                self.action_message = "HAND LOST"
            elif now - self.last_gesture_seen > config.GESTURE_GRACE:
                self.state = GameState.EQUIPPED if hand_seen else GameState.IDLE
                self.action_message = "SCOPE RELEASED"

    def fire(self, default_w: int, default_h: int, now: float) -> bool:
        """
        Attempt to fire a sniper round.
        Returns True if shot was actually fired, False if blocked by state or cooldown.
        """
        if self.state != GameState.SCOPED:
            self.action_message = "BLINK IGNORED - SCOPE FIRST"
            return False

        if self.last_shot_time is not None and (now - self.last_shot_time) < config.FIRE_COOLDOWN:
            self.action_message = "RELOADING"
            return False

        # Fire confirmed
        self.last_shot_time = now
        self.shot_effect_time = now

        shot_x = int(self.aim_x if self.aim_x is not None else default_w // 2)
        shot_y = int(self.aim_y if self.aim_y is not None else default_h // 2)
        shot_center = (shot_x, shot_y)

        hit_idx = self.target_manager.check_hit(float(shot_x), float(shot_y))

        if hit_idx is not None:
            target = self.target_manager.targets[hit_idx]
            target.alive = False
            target.respawn = now + random.uniform(
                config.TARGET_RESPAWN_MIN, config.TARGET_RESPAWN_MAX
            )
            self.stats.record_shot(is_hit=True)
            self.action_message = f"TARGET HIT +{config.TARGET_SCORE_VALUE}"
            self.last_result_label = "HIT!"
            self.last_result_color = config.COLOR_HIT
        else:
            self.stats.record_shot(is_hit=False)
            self.action_message = "SHOT MISSED"
            self.last_result_label = "MISS"
            self.last_result_color = config.COLOR_MISS

        self.last_result_time = now
        self.last_result_center = shot_center
        return True

    def reset(self, width: int, height: int) -> None:
        """Reset gameplay stats, targets, and action message."""
        self.stats.reset()
        self.target_manager.reset(width, height)
        self.aim_x = None
        self.aim_y = None
        self.action_message = "SCORE RESET"
