
import cv2
import mediapipe as mp
import numpy as np
import math
import time

from enum import Enum
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CV SNIPER - STAGE 3
# Hand Tracking | Face Tracking | Sniper Scope | Blink-to-Fire
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

HAND_MODEL = BASE_DIR / "models" / "hand_landmarker.task"
FACE_MODEL = BASE_DIR / "models" / "face_landmarker.task"

CAMERA_INDEX = 0

# Hand activation thresholds
WRIST_RAISED_Y = 0.75
HAND_CENTER_RAISED_Y = 0.65

# Eye blink thresholds (may need calibration)
CLOSED_THRESHOLD = 0.20
OPEN_THRESHOLD = 0.23

MIN_BLINK_MS = 50
MAX_BLINK_MS = 500

FIRE_COOLDOWN_MS = 800
FIRE_DISPLAY_MS = 450

WINDOW_NAME = "CV SNIPER - STAGE 3"


# ============================================================
# GAME STATES
# ============================================================

class GameState(Enum):
    IDLE = 0
    EQUIPPED = 1
    SCOPED = 2


class EyeState(Enum):
    OPEN = 0
    CLOSED = 1


state = GameState.IDLE
eye_state = EyeState.OPEN

blink_start_ms = 0
last_fire_ms = -FIRE_COOLDOWN_MS
last_fire_display_ms = -FIRE_DISPLAY_MS


# ============================================================
# HAND GEOMETRY AND GESTURE DETECTION
# ============================================================

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_is_extended(hand, tip, pip):
    wrist = hand[0]

    return (
        distance(hand[tip], wrist)
        > distance(hand[pip], wrist) * 1.15
    )


def thumb_is_extended(hand):
    wrist = hand[0]

    return (
        distance(hand[4], wrist)
        > distance(hand[3], wrist) * 1.10
    )


def detect_finger_gun(hand):
    index = finger_is_extended(hand, 8, 6)
    middle = finger_is_extended(hand, 12, 10)
    ring = finger_is_extended(hand, 16, 14)
    pinky = finger_is_extended(hand, 20, 18)
    thumb = thumb_is_extended(hand)

    return (
        index
        and thumb
        and not middle
        and not ring
        and not pinky
    )


def get_hand_center(hand):
    return (
        sum(p.x for p in hand) / len(hand),
        sum(p.y for p in hand) / len(hand)
    )


def is_hand_raised(hand):
    # MediaPipe coordinates start at the top-left:
    # smaller Y values are higher on the screen.
    wrist_y = hand[0].y
    _, center_y = get_hand_center(hand)

    return (
        wrist_y < WRIST_RAISED_Y
        or center_y < HAND_CENTER_RAISED_Y
    )


# ============================================================
# FACE / EYE TRACKING
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(face, indices):
    p1, p2, p3, p4, p5, p6 = [
        face[i] for i in indices
    ]

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    return (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)


def get_ear(face):
    left = eye_aspect_ratio(face, LEFT_EYE)
    right = eye_aspect_ratio(face, RIGHT_EYE)

    return (left + right) / 2.0


# ============================================================
# DRAW HAND LANDMARKS
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_hand_landmarks(frame, hand):
    height, width = frame.shape[:2]

    points = [
        (
            int(point.x * width),
            int(point.y * height)
        )
        for point in hand
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    for point in points:
        cv2.circle(
            frame,
            point,
            4,
            (0, 180, 255),
            -1,
            cv2.LINE_AA
        )


# ============================================================
# DRAW FACE LANDMARKS AND EYE OUTLINES
# ============================================================

def draw_face_landmarks(frame, face):
    height, width = frame.shape[:2]

    # All detected facial landmarks
    for landmark in face:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(
                frame,
                (x, y),
                1,
                (255, 200, 0),
                -1
            )

    # Eye outlines
    eye_paths = [
        [33, 160, 158, 133, 153, 144, 33],
        [362, 385, 387, 263, 373, 380, 362]
    ]

    for path in eye_paths:
        points = []

        for index in path:
            landmark = face[index]

            points.append((
                int(landmark.x * width),
                int(landmark.y * height)
            ))

        for i in range(len(points) - 1):
            cv2.line(
                frame,
                points[i],
                points[i + 1],
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )


# ============================================================
# SNIPER SCOPE
# ============================================================

def draw_scope(frame, firing=False):
    height, width = frame.shape[:2]

    cx = width // 2
    cy = height // 2
    radius = max(100, min(width, height) // 3)

    original = frame.copy()

    # Darkened background
    darkened = cv2.convertScaleAbs(
        original,
        alpha=0.35,
        beta=0
    )

    # Circular mask: white inside the scope, black outside.
    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.circle(
        mask,
        (cx, cy),
        radius,
        255,
        -1
    )

    # Original video inside the scope.
    inside = cv2.bitwise_and(
        original,
        original,
        mask=mask
    )

    # Darkened video outside the scope.
    outside = cv2.bitwise_and(
        darkened,
        darkened,
        mask=cv2.bitwise_not(mask)
    )

    frame[:] = cv2.add(inside, outside)

    color = (0, 0, 255) if firing else (0, 255, 0)

    # Outer scope ring
    cv2.circle(
        frame,
        (cx, cy),
        radius,
        color,
        3,
        cv2.LINE_AA
    )

    # Crosshair
    cv2.line(
        frame,
        (cx - radius, cy),
        (cx + radius, cy),
        (255, 255, 255),
        1
    )

    cv2.line(
        frame,
        (cx, cy - radius),
        (cx, cy + radius),
        (255, 255, 255),
        1
    )

    # Inner aiming marks
    gap = 18

    cv2.line(
        frame,
        (cx - 40, cy),
        (cx - gap, cy),
        color,
        2
    )

    cv2.line(
        frame,
        (cx + gap, cy),
        (cx + 40, cy),
        color,
        2
    )

    cv2.line(
        frame,
        (cx, cy - 40),
        (cx, cy - gap),
        color,
        2
    )

    cv2.line(
        frame,
        (cx, cy + gap),
        (cx, cy + 40),
        color,
        2
    )

    cv2.circle(
        frame,
        (cx, cy),
        4,
        color,
        -1
    )

    cv2.putText(
        frame,
        "LIVE SCOPE",
        (cx - 65, max(30, cy - radius - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    if firing:
        cv2.putText(
            frame,
            "FIRE!",
            (cx - 45, min(height - 20, cy + radius + 35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )


# ============================================================
# HUD / DIAGNOSTICS
# ============================================================

def draw_hud(
    frame,
    game_state,
    hand_detected,
    hand_raised,
    finger_gun,
    face_detected,
    eye_status,
    ear,
    action
):
    height, width = frame.shape[:2]

    # HUD background
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (365, 290),
        (10, 10, 10),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.78,
        frame,
        0.22,
        0,
        frame
    )

    state_color = (
        (0, 255, 0)
        if game_state == GameState.SCOPED
        else (0, 220, 255)
    )

    lines = [
        ("CV SNIPER | STAGE 3", (255, 255, 255)),
        ("WEAPON: SNIPER RIFLE", (255, 255, 255)),
        (f"STATE: {game_state.name}", state_color),
        (
            f"HAND DETECTED: {'YES' if hand_detected else 'NO'}",
            (0, 255, 0) if hand_detected else (0, 0, 255)
        ),
        (
            f"HAND RAISED: {'YES' if hand_raised else 'NO'}",
            (0, 255, 0) if hand_raised else (180, 180, 180)
        ),
        (
            f"FINGER GUN: {'YES' if finger_gun else 'NO'}",
            (0, 255, 0) if finger_gun else (180, 180, 180)
        ),
        (
            f"FACE: {'DETECTED' if face_detected else 'NOT FOUND'}",
            (0, 255, 0) if face_detected else (0, 0, 255)
        ),
        (f"EYES: {eye_status}", (255, 255, 255)),
        (f"EAR: {ear:.3f}", (255, 255, 255)),
        (
            f"ACTION: {action}",
            (0, 0, 255) if action == "FIRE"
            else (255, 255, 255)
        )
    ]

    y = 35

    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            color,
            1,
            cv2.LINE_AA
        )
        y += 26

    cv2.putText(
        frame,
        "Q: QUIT",
        (width - 110, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


# ============================================================
# MAIN
# ============================================================

def main():
    global state
    global eye_state
    global blink_start_ms
    global last_fire_ms
    global last_fire_display_ms

    for model_path in (HAND_MODEL, FACE_MODEL):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(HAND_MODEL)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
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
        output_facial_transformation_matrixes=False
    )

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            "Cannot open webcam. Check CAMERA_INDEX."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    previous_timestamp_ms = 0

    print("=" * 55)
    print("CV SNIPER - STAGE 3")
    print("Raise your hand: EQUIPPED")
    print("Make a finger gun: SCOPED")
    print("Blink while scoped: FIRE")
    print("Press Q to quit.")
    print("=" * 55)

    try:
        with (
            vision.HandLandmarker.create_from_options(
                hand_options
            ) as hand_detector,
            vision.FaceLandmarker.create_from_options(
                face_options
            ) as face_detector
        ):
            while True:
                success, frame = camera.read()

                if not success:
                    print("Unable to read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                # Monotonic timestamps for VIDEO mode.
                timestamp_ms = time.monotonic_ns() // 1_000_000

                timestamp_ms = max(
                    timestamp_ms,
                    previous_timestamp_ms + 1
                )

                previous_timestamp_ms = timestamp_ms

                # Run both models.
                hand_result = hand_detector.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                face_result = face_detector.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                # ---------------- HAND ----------------

                hand_detected = bool(hand_result.hand_landmarks)

                hand = None
                finger_gun = False
                hand_raised = False

                if hand_detected:
                    hand = hand_result.hand_landmarks[0]

                    finger_gun = detect_finger_gun(hand)
                    hand_raised = is_hand_raised(hand)

                # ---------------- STATE MACHINE ----------------

                if state == GameState.IDLE:
                    if hand_detected:
                        if finger_gun:
                            # Permit direct entry into scope when
                            # the finger-gun gesture is recognized.
                            state = GameState.SCOPED
                        elif hand_raised:
                            state = GameState.EQUIPPED

                elif state == GameState.EQUIPPED:
                    if not hand_detected:
                        state = GameState.IDLE
                    elif finger_gun:
                        state = GameState.SCOPED

                elif state == GameState.SCOPED:
                    if not hand_detected:
                        state = GameState.IDLE
                    elif not finger_gun:
                        state = GameState.EQUIPPED

                # ---------------- FACE / BLINK ----------------

                face_detected = bool(face_result.face_landmarks)

                ear = 0.0
                blink_event = False

                if face_detected:
                    face = face_result.face_landmarks[0]
                    ear = get_ear(face)

                    if eye_state == EyeState.OPEN:
                        if ear < CLOSED_THRESHOLD:
                            eye_state = EyeState.CLOSED
                            blink_start_ms = timestamp_ms

                    else:
                        if ear > OPEN_THRESHOLD:
                            blink_duration = (
                                timestamp_ms - blink_start_ms
                            )

                            eye_state = EyeState.OPEN

                            if (
                                MIN_BLINK_MS
                                <= blink_duration
                                <= MAX_BLINK_MS
                            ):
                                blink_event = True

                # ---------------- FIRING ----------------

                if (
                    state == GameState.SCOPED
                    and blink_event
                    and timestamp_ms - last_fire_ms
                    >= FIRE_COOLDOWN_MS
                ):
                    last_fire_ms = timestamp_ms
                    last_fire_display_ms = timestamp_ms

                    print(f"[{timestamp_ms}] FIRE EVENT")

                firing = (
                    timestamp_ms - last_fire_display_ms
                    < FIRE_DISPLAY_MS
                )

                action = "FIRE" if firing else "READY"

                # ---------------- RENDERING ----------------
                # Draw scope first, then landmarks and HUD.
                # This keeps the hand and face overlays visible.

                if state == GameState.SCOPED:
                    draw_scope(frame, firing=firing)

                if hand is not None:
                    draw_hand_landmarks(frame, hand)

                if face_detected:
                    draw_face_landmarks(
                        frame,
                        face_result.face_landmarks[0]
                    )

                draw_hud(
                    frame=frame,
                    game_state=state,
                    hand_detected=hand_detected,
                    hand_raised=hand_raised,
                    finger_gun=finger_gun,
                    face_detected=face_detected,
                    eye_status=eye_state.name,
                    ear=ear,
                    action=action
                )

                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("CV Sniper stopped.")


if __name__ == "__main__":
    main()
