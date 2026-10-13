
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


# ============================================================
# CV SNIPER - STAGE 5
# Rounds, difficulty, ammunition, reload, statistics, restart
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
HAND_MODEL = BASE_DIR / "models" / "hand_landmarker.task"
FACE_MODEL = BASE_DIR / "models" / "face_landmarker.task"

CAMERA_INDEX = 0
WINDOW = "CV Sniper - Stage 5"

ROUND_DURATION = 60.0
MAGAZINE_SIZE = 5
RELOAD_DURATION = 1.5
FIRE_COOLDOWN = 0.80

TARGET_COUNT = 4
TARGET_RADIUS = (22, 34)
BASE_TARGET_SPEED = (75, 150)
MAX_TARGET_SPEED_MULTIPLIER = 2.5

AIM_SMOOTHING = 0.30
SCOPE_RADIUS = 155
GESTURE_GRACE = 0.40
HAND_LOST_TIMEOUT = 1.0

CLOSED_EAR = 0.20
MIN_BLINK_DURATION = 0.05
MAX_BLINK_DURATION = 0.50

SHOT_EFFECT_DURATION = 0.18
HIT_MARKER_DURATION = 0.55

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


class RoundState(Enum):
    PLAYING = auto()
    GAME_OVER = auto()


# =========================== DETECTION ===========================

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


# =========================== DRAWING ===========================

def pixel_point(landmark, width, height):
    return (
        int(np.clip(landmark.x * width, 0, width - 1)),
        int(np.clip(landmark.y * height, 0, height - 1)),
    )


def draw_hand(frame, hand):
    h, w = frame.shape[:2]
    points = [pixel_point(lm, w, h) for lm in hand]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (70, 255, 100), 2,
                 cv2.LINE_AA)

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
    outside = cv2.bitwise_and(
        dark, dark, mask=cv2.bitwise_not(mask)
    )
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


# =========================== TARGETS ===========================

def new_target(width, height, difficulty=1.0):
    r = random.randint(*TARGET_RADIUS)
    margin = r + 12

    x = random.randint(margin, max(margin, width - margin - 1))

    low_y = min(height - 1, margin + 60)
    high_y = max(low_y, height - margin - 15)
    y = random.randint(low_y, high_y)

    angle = random.uniform(0, math.tau)
    speed = random.uniform(*BASE_TARGET_SPEED) * difficulty

    return {
        "x": float(x),
        "y": float(y),
        "r": r,
        "vx": math.cos(angle) * speed,
        "vy": math.sin(angle) * speed,
        "alive": True,
        "respawn": 0.0,
    }


def update_targets(targets, width, height, dt, now, difficulty):
    for target in targets:
        if not target["alive"]:
            if now >= target["respawn"]:
                target.update(new_target(width, height, difficulty))
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

        x = int(target["x"])
        y = int(target["y"])
        r = target["r"]
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


# =========================== SHOT EFFECTS ===========================

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


# =========================== HUD / RESULTS ===========================

def draw_hud(frame, state, hand_seen, raised, finger_gun,
             face_seen, eyes, ear, score, hits, shots,
             ammo, reloading, reload_end, round_state,
             time_left, difficulty_level, fps, action, now,
             best_score):

    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (10, 10), (375, 310), (10, 18, 22), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    state_color = {
        GameState.IDLE: (180, 180, 180),
        GameState.EQUIPPED: (0, 220, 255),
        GameState.SCOPED: (70, 255, 100),
    }[state]

    accuracy = (100.0 * hits / shots) if shots else 0.0

    lines = [
        ("CV SNIPER | STAGE 5", (255, 255, 255)),
        (f"STATE: {state.name}", state_color),
        (f"HAND: {hand_seen}  RAISED: {raised}", (220, 240, 240)),
        (f"FINGER GUN: {finger_gun}", (220, 240, 240)),
        (f"FACE: {face_seen}  EYES: {eyes}", (220, 240, 240)),
        (f"EAR: {ear:.3f}", (200, 220, 220)),
        (f"SCORE: {score}  BEST: {best_score}", (80, 255, 110)),
        (f"HITS: {hits}  SHOTS: {shots}  ACC: {accuracy:.1f}%",
         (100, 240, 255)),
        (f"AMMO: {ammo}/{MAGAZINE_SIZE}", (255, 230, 100)),
        (f"TIME: {max(0, time_left):04.1f}s  LEVEL: {difficulty_level}",
         (255, 230, 100)),
        (f"ACTION: {action}", (255, 230, 100)),
        (f"FPS: {fps:.1f}", (180, 190, 190)),
    ]

    y = 34
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.52 if i else 0.57, color, 2 if i == 0 else 1,
            cv2.LINE_AA
        )
        y += 24

    if reloading:
        remaining = max(0.0, reload_end - now)
        cv2.putText(
            frame, f"RELOADING {remaining:.1f}s",
            (w - 230, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 220, 255), 2, cv2.LINE_AA
        )

    cv2.putText(
        frame,
        "Aim with index finger | Blink to fire | R reload | Q quit",
        (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
        0.53, (245, 245, 245), 1, cv2.LINE_AA
    )

    if round_state == RoundState.GAME_OVER:
        panel_w, panel_h = min(540, w - 30), 300
        x1 = (w - panel_w) // 2
        y1 = (h - panel_h) // 2
        x2, y2 = x1 + panel_w, y1 + panel_h

        panel = frame.copy()
        cv2.rectangle(panel, (x1, y1), (x2, y2), (12, 20, 28), -1)
        cv2.addWeighted(panel, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (70, 255, 100), 2, cv2.LINE_AA)

        lines = [
            ("ROUND COMPLETE", (70, 255, 100)),
            (f"FINAL SCORE: {score}", (255, 255, 255)),
            (f"ACCURACY: {accuracy:.1f}%", (100, 240, 255)),
            (f"TARGETS HIT: {hits}", (255, 255, 255)),
            ("Press ENTER to play again", (255, 230, 100)),
            ("Press Q to quit", (200, 200, 200)),
        ]

        y = y1 + 45
        for i, (text, color) in enumerate(lines):
            scale = 0.8 if i == 0 else 0.62
            thickness = 2 if i in (0, 1) else 1
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
            )[0]

            x = (w - text_size[0]) // 2
            cv2.putText(frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, thickness, cv2.LINE_AA)
            y += 42


# =========================== GAME INITIALIZATION ===========================

def reset_round(width, height):
    now = time.monotonic()

    return {
        "start": now,
        "score": 0,
        "hits": 0,
        "shots": 0,
        "ammo": MAGAZINE_SIZE,
        "last_shot": None,
        "shot_time": -10.0,
        "last_blink": -10.0,
        "closed_since": None,
        "last_hand_seen": -10.0,
        "last_gesture_seen": -10.0,
        "last_result_time": -10.0,
        "last_result_center": (0, 0),
        "last_result_label": "",
        "last_result_color": (255, 255, 255),
        "reload_end": 0.0,
        "reloading": False,
        "round_state": RoundState.PLAYING,
        "action": "ROUND STARTED",
        "targets": [
            new_target(width, height) for _ in range(TARGET_COUNT)
        ],
    }


def start_reload(game, now):
    if game["round_state"] != RoundState.PLAYING:
        return

    if game["reloading"]:
        game["action"] = "ALREADY RELOADING"
        return

    if game["ammo"] >= MAGAZINE_SIZE:
        game["action"] = "MAGAZINE FULL"
        return

    game["reloading"] = True
    game["reload_end"] = now + RELOAD_DURATION
    game["action"] = "RELOADING"


def main():
    if not HAND_MODEL.exists():
        raise FileNotFoundError(f"Missing hand model: {HAND_MODEL}")

    if not FACE_MODEL.exists():
        raise FileNotFoundError(f"Missing face model: {FACE_MODEL}")

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(HAND_MODEL)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(FACE_MODEL)
        ),
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
        raise RuntimeError(
            "Cannot open webcam. Close other camera applications."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state = GameState.IDLE
    aim_x = aim_y = None
    last_timestamp_ms = 0
    previous_time = time.monotonic()
    fps = 0.0
    best_score = 0

    ok, first_frame = camera.read()
    if not ok:
        camera.release()
        raise RuntimeError("Could not read the first webcam frame.")

    first_frame = cv2.flip(first_frame, 1)
    height, width = first_frame.shape[:2]
    game = reset_round(width, height)

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
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                hand_result = hand_detector.detect_for_video(
                    mp_image, timestamp_ms
                )
                face_result = face_detector.detect_for_video(
                    mp_image, timestamp_ms
                )

                hand_seen = bool(hand_result.hand_landmarks)
                face_seen = bool(face_result.face_landmarks)

                hand = hand_result.hand_landmarks[0] if hand_seen else None
                face = face_result.face_landmarks[0] if face_seen else None

                raised = hand_is_raised(hand) if hand_seen else False
                finger_gun = detect_finger_gun(hand) if hand_seen else False

                if hand_seen:
                    game["last_hand_seen"] = now
                    draw_hand(frame, hand)

                    raw_x = float(np.clip(hand[8].x, 0, 1)) * w
                    raw_y = float(np.clip(hand[8].y, 0, 1)) * h

                    if aim_x is None:
                        aim_x, aim_y = raw_x, raw_y
                    else:
                        aim_x += (raw_x - aim_x) * AIM_SMOOTHING
                        aim_y += (raw_y - aim_y) * AIM_SMOOTHING

                if finger_gun:
                    game["last_gesture_seen"] = now

                if state == GameState.IDLE:
                    if hand_seen and finger_gun:
                        state = GameState.SCOPED
                        game["action"] = "SCOPE ACQUIRED"
                    elif hand_seen and raised:
                        state = GameState.EQUIPPED
                        game["action"] = "RIFLE EQUIPPED"

                elif state == GameState.EQUIPPED:
                    if finger_gun:
                        state = GameState.SCOPED
                        game["action"] = "SCOPE ACQUIRED"
                    elif now - game["last_hand_seen"] > HAND_LOST_TIMEOUT:
                        state = GameState.IDLE
                        game["action"] = "HAND LOST"

                elif state == GameState.SCOPED:
                    if now - game["last_hand_seen"] > HAND_LOST_TIMEOUT:
                        state = GameState.IDLE
                        game["action"] = "HAND LOST"
                    elif now - game["last_gesture_seen"] > GESTURE_GRACE:
                        state = GameState.EQUIPPED if hand_seen else GameState.IDLE
                        game["action"] = "SCOPE RELEASED"

                # Round timer.
                time_left = ROUND_DURATION - (now - game["start"])

                if time_left <= 0 and game["round_state"] == RoundState.PLAYING:
                    game["round_state"] = RoundState.GAME_OVER
                    game["action"] = "ROUND COMPLETE"
                    best_score = max(best_score, game["score"])

                # Difficulty rises every 300 points, capped at 2.5x.
                difficulty_level = 1 + game["score"] // 300
                difficulty = min(
                    1.0 + 0.15 * (difficulty_level - 1),
                    MAX_TARGET_SPEED_MULTIPLIER
                )

                if game["round_state"] == RoundState.PLAYING:
                    update_targets(
                        game["targets"], w, h, dt, now, difficulty
                    )

                # Face / eye tracking and blink recognition.
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
                        if game["closed_since"] is None:
                            game["closed_since"] = now
                    else:
                        eyes = "OPEN"

                        if game["closed_since"] is not None:
                            duration = now - game["closed_since"]

                            if MIN_BLINK_DURATION <= duration <= MAX_BLINK_DURATION:
                                blink_event = True

                            game["closed_since"] = None
                else:
                    game["closed_since"] = None

                # Complete reload when its timer expires.
                if game["reloading"] and now >= game["reload_end"]:
                    game["ammo"] = MAGAZINE_SIZE
                    game["reloading"] = False
                    game["action"] = "RELOADED"

                # Blink-to-fire: one shot per blink, only when scoped.
                if (
                    blink_event
                    and now - game["last_blink"] > 0.20
                    and game["round_state"] == RoundState.PLAYING
                ):
                    game["last_blink"] = now

                    if state != GameState.SCOPED:
                        game["action"] = "BLINK IGNORED - SCOPE FIRST"

                    elif game["reloading"]:
                        game["action"] = "RELOADING"

                    elif (
                        game["last_shot"] is not None
                        and now - game["last_shot"] < FIRE_COOLDOWN
                    ):
                        game["action"] = "FIRE COOLDOWN"

                    elif game["ammo"] <= 0:
                        game["action"] = "OUT OF AMMO - PRESS R"

                    else:
                        game["last_shot"] = now
                        game["shot_time"] = now
                        game["shots"] += 1
                        game["ammo"] -= 1

                        shot_x = int(aim_x if aim_x is not None else w // 2)
                        shot_y = int(aim_y if aim_y is not None else h // 2)
                        shot_center = (shot_x, shot_y)

                        hit_index = None

                        for i, target in enumerate(game["targets"]):
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
                            target = game["targets"][hit_index]
                            target["alive"] = False
                            target["respawn"] = now + random.uniform(0.7, 1.5)

                            game["score"] += 100
                            game["hits"] += 1
                            game["action"] = "TARGET HIT +100"
                            game["last_result_label"] = "HIT!"
                            game["last_result_color"] = (60, 255, 90)
                        else:
                            game["action"] = "SHOT MISSED"
                            game["last_result_label"] = "MISS"
                            game["last_result_color"] = (40, 80, 255)

                        game["last_result_time"] = now
                        game["last_result_center"] = shot_center
                        best_score = max(best_score, game["score"])

                # Render the scope and target layer.
                aim_center = (
                    int(aim_x if aim_x is not None else w // 2),
                    int(aim_y if aim_y is not None else h // 2),
                )

                recoil = now - game["shot_time"] < SHOT_EFFECT_DURATION

                if state == GameState.SCOPED:
                    recoil_offset = (
                        int(10 * (1 - (now - game["shot_time"]) / SHOT_EFFECT_DURATION))
                        if recoil else 0
                    )

                    scope_center = (
                        aim_center[0],
                        int(np.clip(aim_center[1] + recoil_offset, 0, h - 1))
                    )

                    draw_scope(frame, scope_center, SCOPE_RADIUS, recoil)

                draw_targets(frame, game["targets"])

                if recoil:
                    draw_shot_effect(frame, aim_center, now - game["shot_time"])

                if now - game["last_result_time"] < HIT_MARKER_DURATION:
                    draw_hit_marker(
                        frame,
                        game["last_result_center"],
                        game["last_result_label"],
                        game["last_result_color"]
                    )

                draw_hud(
                    frame=frame,
                    state=state,
                    hand_seen=hand_seen,
                    raised=raised,
                    finger_gun=finger_gun,
                    face_seen=face_seen,
                    eyes=eyes,
                    ear=ear,
                    score=game["score"],
                    hits=game["hits"],
                    shots=game["shots"],
                    ammo=game["ammo"],
                    reloading=game["reloading"],
                    reload_end=game["reload_end"],
                    round_state=game["round_state"],
                    time_left=time_left,
                    difficulty_level=difficulty_level,
                    fps=fps,
                    action=game["action"],
                    now=now,
                    best_score=best_score,
                )

                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                elif key == ord("r"):
                    start_reload(game, time.monotonic())

                elif key in (10, 13) and game["round_state"] == RoundState.GAME_OVER:
                    game = reset_round(w, h)
                    state = GameState.IDLE
                    aim_x = aim_y = None

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
