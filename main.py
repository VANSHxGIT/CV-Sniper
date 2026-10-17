"""
CV Sniper - Stage 4 (Optimized)
Computer Vision gesture-and-eye-controlled sniper simulation game.

Controls:
- Hand Raised: Equip Rifle
- Finger Gun Gesture: Acquire Optic Scope
- Index Finger: Move Aim / Crosshair
- Eye Blink (Natural): Trigger Shot
- 'R': Reset Score and Targets
- 'S': Toggle Landmark Skeletons
- 'D': Toggle HUD Display
- 'Q' / ESC: Quit Game
"""

import sys
import time
from typing import Optional
import cv2

from cv_controller import (
    BlinkDetector,
    GameEngine,
    GameState,
    OptimizedRenderer,
    VisionDetector,
    config,
)


def initialize_camera(
    camera_index: int = config.CAMERA_INDEX,
    width: int = config.FRAME_WIDTH,
    height: int = config.FRAME_HEIGHT,
) -> cv2.VideoCapture:
    """Initialize camera with DirectShow (Windows) or default backend and low latency."""
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError(
            f"Cannot open webcam (Index {camera_index}). Please verify webcam is connected "
            "and not in use by another application."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Crucial optimization: buffer size 1 eliminates frame queue lag
    camera.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)

    return camera


def main() -> None:
    """Main game loop."""
    print("=" * 60)
    print("CV Sniper - Optimized Stage 4")
    print("Initializing MediaPipe detectors and camera...")
    print("=" * 60)

    try:
        detector = VisionDetector()
    except FileNotFoundError as err:
        print(f"\n[ERROR] {err}\n", file=sys.stderr)
        return

    try:
        camera = initialize_camera()
    except RuntimeError as err:
        print(f"\n[ERROR] {err}\n", file=sys.stderr)
        detector.close()
        return

    engine = GameEngine()
    renderer = OptimizedRenderer()
    blink_detector = BlinkDetector()

    show_skeletons = True
    show_hud = True

    fps = 0.0
    previous_time = time.monotonic()
    last_timestamp_ms = 0

    print("Camera and detectors ready. Launching game window...")
    print("Controls: Aim with index finger, blink to fire, 'R' to reset, 'Q' to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read webcam frame.")
                break

            # Mirror the frame horizontally for natural intuitive aiming
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.monotonic()
            dt = min(now - previous_time, 0.05)
            previous_time = now

            if dt > 0:
                instant_fps = 1.0 / dt
                fps = instant_fps if fps == 0.0 else fps * 0.9 + instant_fps * 0.1

            # Ensure monotonically increasing timestamp for MediaPipe VIDEO mode
            timestamp_ms = max(int(now * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            # 1. Run vision detection
            vision_data = detector.detect(frame, timestamp_ms)

            # 2. Update player aim position if hand detected
            if vision_data.aim_point is not None:
                engine.update_aim(vision_data.aim_point[0], vision_data.aim_point[1])

            # 3. Update game state machine
            engine.update_states(
                hand_seen=vision_data.hand_seen,
                hand_raised=vision_data.hand_raised,
                finger_gun=vision_data.finger_gun,
                now=now,
            )

            # 4. Update target simulation & physics
            engine.target_manager.update(w, h, dt, now)

            # 5. Process eye blink detection
            blink_event, eye_state = blink_detector.process(
                vision_data.ear if vision_data.face_seen else None,
                now=now,
            )

            # 6. Execute trigger pull if valid blink occurred
            if blink_event:
                engine.fire(default_w=w, default_h=h, now=now)

            # ===================== RENDERING =====================
            # Draw skeletons if enabled
            if show_skeletons:
                if vision_data.hand_seen and vision_data.hand is not None:
                    renderer.draw_hand(frame, vision_data.hand)
                if vision_data.face_seen and vision_data.face is not None:
                    renderer.draw_face_eyes(frame, vision_data.face)

            # Compute smoothed crosshair coordinates
            aim_cx = int(engine.aim_x if engine.aim_x is not None else w // 2)
            aim_cy = int(engine.aim_y if engine.aim_y is not None else h // 2)

            recoil_active = (now - engine.shot_effect_time) < config.SHOT_EFFECT_DURATION

            # Draw sniper scope when SCOPED
            if engine.state == GameState.SCOPED:
                recoil_offset = 0
                if recoil_active:
                    recoil_progress = (now - engine.shot_effect_time) / config.SHOT_EFFECT_DURATION
                    recoil_offset = int(10 * (1.0 - recoil_progress))

                scope_center = (aim_cx, min(max(aim_cy + recoil_offset, 0), h - 1))
                renderer.draw_scope(frame, scope_center, config.SCOPE_RADIUS, recoil=recoil_active)

            # Targets are drawn after scope to ensure visibility
            renderer.draw_targets(frame, engine.target_manager.targets)

            # Draw muzzle flash recoil effect
            if recoil_active:
                renderer.draw_shot_effect(frame, (aim_cx, aim_cy), now - engine.shot_effect_time)

            # Draw hit / miss marker
            if (now - engine.last_result_time) < config.HIT_MARKER_DURATION:
                renderer.draw_hit_marker(
                    frame,
                    engine.last_result_center,
                    engine.last_result_label,
                    engine.last_result_color,
                )

            # Draw HUD overlay
            if show_hud:
                renderer.draw_hud(
                    frame=frame,
                    state=engine.state,
                    hand_seen=vision_data.hand_seen,
                    raised=vision_data.hand_raised,
                    finger_gun=vision_data.finger_gun,
                    face_seen=vision_data.face_seen,
                    eyes=eye_state,
                    ear=vision_data.ear,
                    score=engine.stats.score,
                    hits=engine.stats.hits,
                    shots=engine.stats.shots,
                    action=engine.action_message,
                    fps=fps,
                    last_shot=engine.last_shot_time,
                    now=now,
                )

            cv2.imshow(config.WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord("r"):      # Reset
                engine.reset(w, h)
            elif key == ord("s"):      # Toggle skeleton
                show_skeletons = not show_skeletons
            elif key == ord("d"):      # Toggle HUD
                show_hud = not show_hud

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()
        print("Game shut down cleanly.")


if __name__ == "__main__":
    main()
