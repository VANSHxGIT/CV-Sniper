"""Comprehensive automated test suite for CV Sniper."""

import math
import unittest
from types import SimpleNamespace
import numpy as np

import cv_controller.config as config
from cv_controller.game import GameEngine, GameState, GameStats, Target, TargetManager
from cv_controller.gestures import (
    BlinkDetector,
    detect_finger_gun,
    distance,
    eye_aspect_ratio,
    finger_extended,
    hand_is_raised,
)
from cv_controller.renderer import OptimizedRenderer, pixel_point


class TestGestures(unittest.TestCase):
    """Test geometric and gesture detection algorithms."""

    def test_distance(self):
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        self.assertAlmostEqual(distance(p1, p2), 5.0)

        # Test with landmark-like objects
        lm1 = SimpleNamespace(x=1.0, y=1.0)
        lm2 = SimpleNamespace(x=4.0, y=5.0)
        self.assertAlmostEqual(distance(lm1, lm2), 5.0)

    def test_finger_gun_detection(self):
        # Create a mock hand landmark list (21 landmarks)
        # Wrist at (0.5, 0.9)
        hand = [SimpleNamespace(x=0.5, y=0.9, z=0.0) for _ in range(21)]

        # Index finger extended upwards: tip (8) far from wrist (0), pip (6) closer
        hand[6] = SimpleNamespace(x=0.5, y=0.6, z=0.0)
        hand[8] = SimpleNamespace(x=0.5, y=0.3, z=0.0)

        # Thumb extended outwards
        hand[3] = SimpleNamespace(x=0.4, y=0.8, z=0.0)
        hand[4] = SimpleNamespace(x=0.3, y=0.75, z=0.0)

        # Middle, ring, pinky curled in
        hand[10] = SimpleNamespace(x=0.52, y=0.65, z=0.0)
        hand[12] = SimpleNamespace(x=0.52, y=0.70, z=0.0)  # curled: tip closer to wrist than extended

        hand[14] = SimpleNamespace(x=0.54, y=0.65, z=0.0)
        hand[16] = SimpleNamespace(x=0.54, y=0.70, z=0.0)

        hand[18] = SimpleNamespace(x=0.56, y=0.65, z=0.0)
        hand[20] = SimpleNamespace(x=0.56, y=0.70, z=0.0)

        self.assertTrue(detect_finger_gun(hand))

    def test_hand_is_raised(self):
        # High hand (wrist_y = 0.5 < 0.75)
        hand_high = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        self.assertTrue(hand_is_raised(hand_high))

        # Low hand (wrist_y = 0.9 >= 0.75 and palm_y = 0.9 >= 0.65)
        hand_low = [SimpleNamespace(x=0.5, y=0.9, z=0.0) for _ in range(21)]
        self.assertFalse(hand_is_raised(hand_low))

    def test_eye_aspect_ratio(self):
        # 6 eye points
        face = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(468)]
        indices = (0, 1, 2, 3, 4, 5)

        # Horizontal points (p1 and p4)
        face[0] = SimpleNamespace(x=0.1, y=0.5)
        face[3] = SimpleNamespace(x=0.3, y=0.5)  # horizontal = 0.2

        # Vertical points (p2 to p6, p3 to p5)
        face[1] = SimpleNamespace(x=0.15, y=0.45)
        face[5] = SimpleNamespace(x=0.15, y=0.55)  # vert1 = 0.10

        face[2] = SimpleNamespace(x=0.25, y=0.45)
        face[4] = SimpleNamespace(x=0.25, y=0.55)  # vert2 = 0.10

        # EAR = (0.10 + 0.10) / (2 * 0.20) = 0.50
        ear = eye_aspect_ratio(face, indices)
        self.assertAlmostEqual(ear, 0.50)

    def test_blink_detector_lifecycle(self):
        detector = BlinkDetector(
            closed_ear=0.20,
            open_ear=0.235,
            min_duration=0.05,
            max_duration=0.50,
            cooldown=0.20,
        )

        # Eye initially open
        event, state = detector.process(0.30, now=1.0)
        self.assertFalse(event)
        self.assertEqual(state, "OPEN")

        # Eye closes
        event, state = detector.process(0.15, now=1.05)
        self.assertFalse(event)
        self.assertEqual(state, "CLOSED")

        # Eye stays closed for 0.15s (valid blink window)
        event, state = detector.process(0.15, now=1.15)
        self.assertFalse(event)
        self.assertEqual(state, "CLOSED")

        # Eye re-opens at 1.20s (duration = 0.15s) -> BLINK FIRES!
        event, state = detector.process(0.30, now=1.20)
        self.assertTrue(event)
        self.assertEqual(state, "OPEN")

        # Subsequent open frame does NOT fire again
        event, state = detector.process(0.30, now=1.22)
        self.assertFalse(event)

        # Cooldown check: closing and reopening immediately within cooldown should not trigger
        detector.process(0.15, now=1.25)
        event, _ = detector.process(0.30, now=1.35)  # 1.35 - 1.20 < 0.20 cooldown
        self.assertFalse(event)


class TestGameEngine(unittest.TestCase):
    """Test target simulation, hit mechanics, and game state machine."""

    def test_target_creation_and_bounds(self):
        w, h = 1280, 720
        target = Target.create_random(w, h)
        self.assertTrue(target.alive)
        self.assertGreaterEqual(target.x, target.r)
        self.assertLessEqual(target.x, w - target.r)
        self.assertGreater(target.r, 0)

    def test_target_physics_bounce(self):
        w, h = 1000, 500
        # Target moving left towards edge
        target = Target(x=15.0, y=200.0, r=20, vx=-100.0, vy=0.0, alive=True)
        target.update(w, h, dt=0.2, now=1.0)
        # Should have bounced and reversed vx
        self.assertGreaterEqual(target.vx, 0)
        self.assertGreaterEqual(target.x, target.r)

    def test_target_hit_detection(self):
        target = Target(x=500.0, y=300.0, r=30, vx=0.0, vy=0.0, alive=True)

        # Direct center hit
        self.assertTrue(target.is_hit(500.0, 300.0))

        # Edge hit within radius + tolerance
        self.assertTrue(target.is_hit(500.0 + 35.0, 300.0))

        # Far miss
        self.assertFalse(target.is_hit(700.0, 300.0))

    def test_stats_tracking(self):
        stats = GameStats()
        self.assertEqual(stats.accuracy, 0.0)

        stats.record_shot(is_hit=True)
        self.assertEqual(stats.shots, 1)
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.score, 100)
        self.assertEqual(stats.accuracy, 100.0)

        stats.record_shot(is_hit=False)
        self.assertEqual(stats.shots, 2)
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.accuracy, 50.0)

        stats.reset()
        self.assertEqual(stats.score, 0)
        self.assertEqual(stats.shots, 0)

    def test_engine_state_transitions(self):
        engine = GameEngine()
        self.assertEqual(engine.state, GameState.IDLE)

        now = 10.0
        # Raise hand -> EQUIPPED
        engine.update_states(hand_seen=True, hand_raised=True, finger_gun=False, now=now)
        self.assertEqual(engine.state, GameState.EQUIPPED)

        # Finger gun -> SCOPED
        now += 0.1
        engine.update_states(hand_seen=True, hand_raised=True, finger_gun=True, now=now)
        self.assertEqual(engine.state, GameState.SCOPED)

        # Release finger gun for longer than GESTURE_GRACE -> EQUIPPED
        now += config.GESTURE_GRACE + 0.1
        engine.update_states(hand_seen=True, hand_raised=True, finger_gun=False, now=now)
        self.assertEqual(engine.state, GameState.EQUIPPED)

        # Lose hand for longer than HAND_LOST_TIMEOUT -> IDLE
        now += config.HAND_LOST_TIMEOUT + 0.1
        engine.update_states(hand_seen=False, hand_raised=False, finger_gun=False, now=now)
        self.assertEqual(engine.state, GameState.IDLE)


class TestRenderer(unittest.TestCase):
    """Test OpenCV rendering routines for memory stability and bounds checking."""

    def setUp(self):
        self.renderer = OptimizedRenderer(scope_radius=155)
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_scope_clipping_corners(self):
        # Render scope at center and all four screen corners
        test_centers = [
            (640, 360),
            (0, 0),
            (1280, 0),
            (0, 720),
            (1280, 720),
            (-50, -50),
            (1500, 900),
        ]
        for cx, cy in test_centers:
            frame = self.frame.copy()
            self.renderer.draw_scope(frame, (cx, cy))
            self.assertEqual(frame.shape, (720, 1280, 3))

    def test_hud_rendering(self):
        frame = self.frame.copy()
        self.renderer.draw_hud(
            frame=frame,
            state=GameState.SCOPED,
            hand_seen=True,
            raised=True,
            finger_gun=True,
            face_seen=True,
            eyes="OPEN",
            ear=0.28,
            score=500,
            hits=5,
            shots=6,
            action="TARGET HIT +100",
            fps=59.4,
            last_shot=time.monotonic() - 0.2,
            now=time.monotonic(),
        )
        self.assertEqual(frame.shape, (720, 1280, 3))

    def test_shot_effect_rendering(self):
        frame = self.frame.copy()
        self.renderer.draw_shot_effect(frame, (640, 360), elapsed=0.05)
        self.assertEqual(frame.shape, (720, 1280, 3))

    def test_pixel_point_clamping(self):
        lm_oob = SimpleNamespace(x=1.5, y=-0.5)
        pt = pixel_point(lm_oob, 1280, 720)
        self.assertEqual(pt[0], 1279)
        self.assertEqual(pt[1], 0)


if __name__ == "__main__":
    unittest.main()
