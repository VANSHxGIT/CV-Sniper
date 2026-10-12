
import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
from enum import Enum, auto
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ===================== CONFIGURATION =====================

BASE_DIR = Path(__file__).resolve().parent
HAND_MODEL = BASE_DIR / "models" / "hand_landmarker.task"
FACE_MODEL = BASE_DIR / "models" / "face_landmarker.task"

CAMERA_INDEX = 0
WINDOW = "CV Sniper - Stage 4"

SCOPE_RADIUS = 155
TARGET_COUNT = 4
TARGET_RADIUS = (22, 34)
TARGET_SPEED = (75, 150)

AIM_SMOOTHING = 0.30
FIRE_COOLDOWN = 0.80       # seconds
SHOT_EFFECT_DURATION = 0.18
HIT_MARKER_DURATION = 0.55
GESTURE_GRACE = 0.40
HAND_LOST_TIMEOUT = 1.0

CLOSED_EAR = 0.20
OPEN_EAR = 0.235
MIN_BLINK_DURATION = 0.05
MAX_BLINK_DURATION = 0.50

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class GameState(Enum):
    IDLE = auto()
    EQUIPPED = auto()
    SCOPED = auto()


# ===================== GESTURE / EYE DETECTION =====================

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_extended(hand, tip, pip):
    wrist = hand[0]
    return distance(hand[tip], wrist) > distance(hand[pip], wrist) * 1.15


def detect_finger_gun(hand):
    index = finger_extended(hand, 8, 6)
    middle = finger_extended(hand, 12, 10)
    ring = finger_extended(hand, 16, 14)
    pinky = finger_extended(hand, 20, 18)
    thumb = distance(hand[4], hand[0]) > distance(hand[3], hand[0]) * 1.10

    return index and thumb and not middle and not ring and not pinky


def hand_is_raised(hand):
    wrist_y = hand[0].y
    palm_y = sum(hand[i].y for i in (0, 5, 9, 13, 17)) / 5
    return wrist_y < 0.75 or palm_y < 0.65


def eye_aspect_ratio(face, indices):
    p1, p2, p3, p4, p5, p6 = [face[i] for i in indices]
    horizontal = distance(p1, p4)

    if horizontal < 1e-6:
        return 1.0

    return (
        distance(p2, p6) + distance(p3, p5)
    ) / (2.0 * horizontal)


# ===================== DRAWING =====================

def pixel_point(landmark, width, height):
    return (
        int(np.clip(landmark.x * width, 0, width - 1)),
        int(np.clip(landmark.y * height, 0, height - 1)),
    )


def draw_hand(frame, hand):
    h, w = frame.shape[:2]
    points = [pixel_point(lm, w, h) for lm in hand]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b],
                 (70, 255, 100), 2, cv2.LINE_AA)

    for point in points:
        cv2.circle(frame, point, 4, (40, 230, 255), -1, cv2.LINE_AA)


def draw_face_eyes(frame, face):
    h, w = frame.shape[:2]

    for indices in (LEFT_EYE, RIGHT_EYE):
        points = np.array(
            [pixel_point(face[i], w, h) for i in indices],
            dtype=np.int32
        )
        cv2.polylines(frame, [points], True, (255, 220, 60), 2,
                      cv2.LINE_AA)

        for x, y in points:
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1,
                       cv2.LINE_AA)


def draw_scope(frame, center, radius, recoil=False):
    h, w = frame.shape[:2]
    cx = int(np.clip(center[0], 0, w - 1))
    cy = int(np.clip(center[1], 0, h - 1))

    original = frame.copy()
    dark = cv2.convertScaleAbs(original, alpha=0.35, beta=0)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    inside = cv2.bitwise_and(original, original, mask=mask)
    outside = cv2.bitwise_and(dark, dark, mask=cv2.bitwise_not(mask))
    frame[:] = cv2.add(inside, outside)

    cv2.circle(frame, (cx, cy), radius, (220, 235, 220), 3,
               cv2.LINE_AA)

    color = (30, 40, 255) if recoil else (50, 255, 90)
    gap, length = 12, 38

    cv2.line(frame, (cx - length, cy), (cx - gap, cy),
             color, 2, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy), (cx + length, cy),
             color, 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - length), (cx, cy - gap),
             color, 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap), (cx, cy + length),
             color, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)

    for angle_deg in range(0, 360, 30):
        angle = math.radians(angle_deg)
        r1, r2 = radius - 13, radius - 4
        p1 = (int(cx + math.cos(angle) * r1),
              int(cy + math.sin(angle) * r1))
        p2 = (int(cx + math.cos(angle) * r2),
              int(cy + math.sin(angle) * r2))
        cv2.line(frame, p1, p2, (220, 235, 220), 1, cv2.LINE_AA)


def new_target(width, height):
    r = random.randint(*TARGET_RADIUS)
    margin = r + 12
    x = random.randint(margin, max(margin, width - margin - 1))
    y_min = min(height - 1, margin + 60)
    y_max = max(y_min, height - margin - 15)
    y = random.randint(y_min, y_max)

    angle = random.uniform(0, math.tau)
    speed = random.uniform(*TARGET_SPEED)

    return {
        "x": float(x), "y": float(y), "r": r,
        "vx": math.cos(angle) * speed,
        "vy": math.sin(angle) * speed,
        "alive": True, "respawn": 0.0
    }


def update_targets(targets, width, height, dt, now):
    for target in targets:
        if not target["alive"]:
            if now >= target["respawn"]:
                target.update(new_target(width, height))
            continue

        target["x"] += target["vx"] * dt
        target["y"] += target["vy"] * dt
        r = target["r"]

        if target["x"] < r or target["x"] > width - r:
            target["vx"] *= -1
            target["x"] = float(np.clip(target["x"], r, width - r))

        low_y = min(height - r, r + 45)
        high_y = max(low_y, height - r - 10)

        if target["y"] < low_y or target["y"] > high_y:
            target["vy"] *= -1
            target["y"] = float(np.clip(target["y"], low_y, high_y))


def draw_targets(frame, targets):
    for target in targets:
        if not target["alive"]:
            continue

        x, y, r = int(target["x"]), int(target["y"]), target["r"]
        color = (60, 220, 255)

        cv2.circle(frame, (x, y), r, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), int(r * 0.62), color, 2,
                   cv2.LINE_AA)
        cv2.circle(frame, (x, y), max(3, int(r * 0.18)),
                   color, -1, cv2.LINE_AA)
        cv2.line(frame, (x - r - 7, y), (x + r + 7, y),
                 color, 1, cv2.LINE_AA)
        cv2.line(frame, (x, y - r - 7), (x, y + r + 7),
                 color, 1, cv2.LINE_AA)


def draw_shot_effect(frame, center, elapsed):
    if not 0 <= elapsed < SHOT_EFFECT_DURATION:
        return

    cx, cy = map(int, center)
    progress = elapsed / SHOT_EFFECT_DURATION
    radius = int(35 * (1 - progress)) + 8

    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius, (80, 220, 255),
               -1, cv2.LINE_AA)

    for angle_deg in range(0, 360, 45):
        angle = math.radians(angle_deg)
        p1 = (int(cx + math.cos(angle) * (radius + 5)),
              int(cy + math.sin(angle) * (radius + 5)))
        p2 = (int(cx + math.cos(angle) * (radius + 25)),
              int(cy + math.sin(angle) * (radius + 25)))
        cv2.line(overlay, p1, p2, (100, 245, 255), 3, cv2.LINE_AA)

    alpha = 0.45 * (1 - progress)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_hit_marker(frame, center, label, color):
    cx, cy = map(int, center)
    size = 15

    for sx in (-1, 1):
        for sy in (-1, 1):
            cv2.line(
                frame,
                (cx + sx * size, cy + sy * size),
                (cx + sx * 5, cy + sy * 5),
                color, 3, cv2.LINE_AA
            )

    cv2.putText(frame, label, (cx + 22, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                cv2.LINE_AA)


def draw_hud(frame, state, hand_seen, raised, finger_gun,
             face_seen, eyes, ear, score, hits, shots, action, fps,
             last_shot, now):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (345, 225), (10, 18, 22), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    state_color = {
        GameState.IDLE: (180, 180, 180),
        GameState.EQUIPPED: (0, 220, 255),
        GameState.SCOPED: (70, 255, 100)
    }[state]

    lines = [
        ("CV SNIPER | STAGE 4", (255, 255, 255)),
        (f"STATE: {state.name}", state_color),
        (f"HAND: {hand_seen}  RAISED: {raised}", (220, 240, 240)),
        (f"FINGER GUN: {finger_gun}", (220, 240, 240)),
        (f"FACE: {face_seen}  EYES: {eyes}", (220, 240, 240)),
        (f"EAR: {ear:.3f}", (200, 220, 220)),
        (f"SCORE: {score}  HITS: {hits}/{shots}", (80, 255, 110)),
        (f"ACTION: {action}", (255, 230, 100)),
        (f"FPS: {fps:.1f}", (180, 190, 190)),
    ]

    y = 35
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.53 if i else 0.57, color, 2 if i == 0 else 1,
                    cv2.LINE_AA)
        y += 23

    if state == GameState.IDLE:
        instruction = "Raise hand or show finger-gun to start"
    elif state == GameState.EQUIPPED:
        instruction = "Make finger-gun gesture to scope in"
    else:
        instruction = "Aim with index finger; blink to fire"

    cv2.putText(frame, instruction, (15, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245),
                2, cv2.LINE_AA)

    if last_shot is not None:
        remaining = FIRE_COOLDOWN - (now - last_shot)
        if remaining > 0:
            cv2.putText(frame, f"RELOAD {remaining:.1f}s",
                        (w - 170, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 220, 255), 2, cv2.LINE_AA)


# ===================== MAIN GAME LOOP =====================

def main():
    if not HAND_MODEL.exists():
        raise FileNotFoundError(f"Missing hand model: {HAND_MODEL}")
    if not FACE_MODEL.exists():
        raise FileNotFoundError(f"Missing face model: {FACE_MODEL}")

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Cannot open webcam. Close other camera apps.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state = GameState.IDLE
    score = hits = shots = 0
    targets = []
    aim_x = aim_y = None

    last_shot = None
    shot_time = -10.0
    last_result_time = -10.0
    last_result_center = (0, 0)
    last_result_label = ""
    last_result_color = (255, 255, 255)

    closed_since = None
    last_blink = -10.0
    last_hand_seen = -10.0
    last_gesture_seen = -10.0

    action = "READY"
    previous_time = time.monotonic()
    last_timestamp_ms = 0
    fps = 0.0

    try:
        with (
            vision.HandLandmarker.create_from_options(hand_options) as hand_detector,
            vision.FaceLandmarker.create_from_options(face_options) as face_detector
        ):
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read webcam frame.")
                    break

                # Mirror the image for natural hand-controlled aiming.
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                now = time.monotonic()
                dt = min(now - previous_time, 0.05)
                previous_time = now

                if dt > 0:
                    instant_fps = 1.0 / dt
                    fps = instant_fps if fps == 0 else fps * 0.9 + instant_fps * 0.1

                timestamp_ms = max(int(now * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
                face_result = face_detector.detect_for_video(mp_image, timestamp_ms)

                hand_seen = bool(hand_result.hand_landmarks)
                face_seen = bool(face_result.face_landmarks)
                hand = hand_result.hand_landmarks[0] if hand_seen else None
                face = face_result.face_landmarks[0] if face_seen else None

                raised = hand_is_raised(hand) if hand_seen else False
                finger_gun = detect_finger_gun(hand) if hand_seen else False

                if hand_seen:
                    last_hand_seen = now
                    draw_hand(frame, hand)

                    # Index fingertip controls the crosshair and scope.
                    raw_x = float(np.clip(hand[8].x, 0, 1)) * w
                    raw_y = float(np.clip(hand[8].y, 0, 1)) * h

                    if aim_x is None:
                        aim_x, aim_y = raw_x, raw_y
                    else:
                        aim_x += (raw_x - aim_x) * AIM_SMOOTHING
                        aim_y += (raw_y - aim_y) * AIM_SMOOTHING

                if finger_gun:
                    last_gesture_seen = now

                # State transitions; short gesture gaps won't instantly drop scope.
                if state == GameState.IDLE:
                    if hand_seen and finger_gun:
                        state = GameState.SCOPED
                        action = "SCOPE ACQUIRED"
                    elif hand_seen and raised:
                        state = GameState.EQUIPPED
                        action = "RIFLE EQUIPPED"

                elif state == GameState.EQUIPPED:
                    if finger_gun:
                        state = GameState.SCOPED
                        action = "SCOPE ACQUIRED"
                    elif now - last_hand_seen > HAND_LOST_TIMEOUT:
                        state = GameState.IDLE
                        action = "HAND LOST"

                elif state == GameState.SCOPED:
                    if now - last_hand_seen > HAND_LOST_TIMEOUT:
                        state = GameState.IDLE
                        action = "HAND LOST"
                    elif now - last_gesture_seen > GESTURE_GRACE:
                        state = GameState.EQUIPPED if hand_seen else GameState.IDLE
                        action = "SCOPE RELEASED"

                if not targets:
                    targets = [new_target(w, h) for _ in range(TARGET_COUNT)]

                update_targets(targets, w, h, dt, now)

                # Blink detection: register on reopening, not continuously while closed.
                ear = 1.0
                eyes = "UNKNOWN"
                blink_event = False

                if face_seen:
                    draw_face_eyes(frame, face)
                    ear = (
                        eye_aspect_ratio(face, LEFT_EYE)
                        + eye_aspect_ratio(face, RIGHT_EYE)
                    ) / 2.0

                    if ear < CLOSED_EAR:
                        eyes = "CLOSED"
                        if closed_since is None:
                            closed_since = now
                    else:
                        eyes = "OPEN"
                        if closed_since is not None:
                            duration = now - closed_since
                            if MIN_BLINK_DURATION <= duration <= MAX_BLINK_DURATION:
                                blink_event = True
                            closed_since = None
                else:
                    closed_since = None

                # One shot per valid blink, only while scoped and off cooldown.
                if blink_event and now - last_blink > 0.20:
                    last_blink = now

                    if state != GameState.SCOPED:
                        action = "BLINK IGNORED - SCOPE FIRST"
                    elif last_shot is not None and now - last_shot < FIRE_COOLDOWN:
                        action = "RELOADING"
                    else:
                        last_shot = now
                        shot_time = now
                        shots += 1

                        shot_x = int(aim_x if aim_x is not None else w // 2)
                        shot_y = int(aim_y if aim_y is not None else h // 2)
                        shot_center = (shot_x, shot_y)

                        hit_index = None
                        for i, target in enumerate(targets):
                            if not target["alive"]:
                                continue

                            d = math.hypot(
                                shot_x - target["x"],
                                shot_y - target["y"]
                            )
                            if d <= target["r"] + 8:
                                hit_index = i
                                break

                        if hit_index is not None:
                            target = targets[hit_index]
                            target["alive"] = False
                            target["respawn"] = now + random.uniform(0.7, 1.5)
                            score += 100
                            hits += 1
                            action = "TARGET HIT +100"
                            last_result_label = "HIT!"
                            last_result_color = (60, 255, 90)
                        else:
                            action = "SHOT MISSED"
                            last_result_label = "MISS"
                            last_result_color = (40, 80, 255)

                        last_result_time = now
                        last_result_center = shot_center

                # Scope follows the smoothed fingertip.
                aim_center = (
                    int(aim_x if aim_x is not None else w // 2),
                    int(aim_y if aim_y is not None else h // 2),
                )
                recoil = now - shot_time < SHOT_EFFECT_DURATION

                if state == GameState.SCOPED:
                    recoil_offset = int(10 * (1 - (now - shot_time) / SHOT_EFFECT_DURATION)) if recoil else 0
                    scope_center = (
                        aim_center[0],
                        int(np.clip(aim_center[1] + recoil_offset, 0, h - 1))
                    )
                    draw_scope(frame, scope_center, SCOPE_RADIUS, recoil)

                # Targets are drawn after the scope so they remain visible.
                draw_targets(frame, targets)

                if recoil:
                    draw_shot_effect(frame, aim_center, now - shot_time)

                if now - last_result_time < HIT_MARKER_DURATION:
                    draw_hit_marker(
                        frame, last_result_center,
                        last_result_label, last_result_color
                    )

                draw_hud(
                    frame, state, hand_seen, raised, finger_gun,
                    face_seen, eyes, ear, score, hits, shots,
                    action, fps, last_shot, now
                )

                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    score = hits = shots = 0
                    targets = [new_target(w, h) for _ in range(TARGET_COUNT)]
                    action = "SCORE RESET"

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
