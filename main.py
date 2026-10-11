
import cv2
import mediapipe as mp
import math
import time

from enum import Enum
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CV SNIPER - STAGE 3
# Hand Tracking + Face Tracking + Scope + Blink-to-Fire
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
FACE_MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

CAMERA_INDEX = 0
HAND_RAISED_THRESHOLD = 0.55

CLOSED_THRESHOLD = 0.20
OPEN_THRESHOLD = 0.23

MIN_BLINK_DURATION_MS = 50
MAX_BLINK_DURATION_MS = 500

FIRE_DISPLAY_MS = 450
FIRE_COOLDOWN_MS = 800

WINDOW_NAME = "CV SNIPER | Stage 3"


# -------------------- GAME STATES ---------------------------

class GameState(Enum):
    IDLE = 0
    EQUIPPED = 1
    SCOPED = 2


class EyeState(Enum):
    OPEN = 0
    CLOSED = 1


# -------------------- RUNTIME STATE -------------------------

state = GameState.IDLE
eye_state = EyeState.OPEN

blink_start_time = 0
last_fire_time = -FIRE_COOLDOWN_MS
last_fire_display_time = -FIRE_DISPLAY_MS


# -------------------- GEOMETRY ------------------------------

def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


def finger_is_extended(landmarks, tip, pip):
    wrist = landmarks[0]

    return (
        distance(landmarks[tip], wrist)
        > distance(landmarks[pip], wrist) * 1.15
    )


def thumb_is_extended(landmarks):
    wrist = landmarks[0]

    return (
        distance(landmarks[4], wrist)
        > distance(landmarks[3], wrist) * 1.10
    )


def detect_finger_gun(hand):
    index = finger_is_extended(hand, 8, 6)
    middle = finger_is_extended(hand, 12, 10)
    ring = finger_is_extended(hand, 16, 14)
    pinky = finger_is_extended(hand, 20, 18)
    thumb = thumb_is_extended(hand)

    return (
        index and thumb
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
    _, center_y = get_hand_center(hand)
    return center_y < HAND_RAISED_THRESHOLD


# -------------------- EYE TRACKING --------------------------

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(face_landmarks, indices):
    p1, p2, p3, p4, p5, p6 = [
        face_landmarks[i] for i in indices
    ]

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    return (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)


def get_eye_aspect_ratio(face_landmarks):
    left = eye_aspect_ratio(face_landmarks, LEFT_EYE)
    right = eye_aspect_ratio(face_landmarks, RIGHT_EYE)

    return (left + right) / 2.0


# -------------------- HAND DRAWING --------------------------

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
        (int(p.x * width), int(p.y * height))
        for p in hand
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
            frame, point, 3,
            (0, 180, 255), -1,
            cv2.LINE_AA
        )


# -------------------- FACE DRAWING -------------------------

def draw_face_landmarks(frame, face_landmarks):
    height, width = frame.shape[:2]

    # Draw all detected face landmarks.
    for landmark in face_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(
                frame, (x, y), 1,
                (255, 200, 0), -1
            )

    # Highlight both eyes.
    eye_paths = [
        [33, 160, 158, 133, 153, 144, 33],
        [362, 385, 387, 263, 373, 380, 362]
    ]

    for path in eye_paths:
        points = []

        for index in path:
            landmark = face_landmarks[index]
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

    cv2.putText(
        frame,
        "FACE TRACKING: ON",
        (width - 260, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 200, 0),
        2
    )


# -------------------- SCOPE DRAWING -------------------------

def draw_scope(frame, firing=False):
    height, width = frame.shape[:2]

    center_x = width // 2
    center_y = height // 2

    radius = max(100, min(width, height) // 3)

    original = frame.copy()

    # Darken the whole image.
    darkened = cv2.convertScaleAbs(
        original,
        alpha=0.35,
        beta=0
    )

    # Build a circular mask.
    mask = cv2.cvtColor(
        cv2.UMat(height, width, cv2.CV_8UC1).get(),
        cv2.COLOR_GRAY2BGR
    )

    mask[:] = 0

    cv2.circle(
        mask,
        (center_x, center_y),
        radius,
        (255, 255, 255),
        -1
    )

    mask = cv2.cvtColor(
        mask,
        cv2.COLOR_BGR2GRAY
    )

    # Preserve the original camera feed inside the scope.
    inside = cv2.bitwise_and(
        original, original, mask=mask
    )

    outside_mask = cv2.bitwise_not(mask)

    outside = cv2.bitwise_and(
        darkened, darkened, mask=outside_mask
    )

    frame[:] = cv2.add(inside, outside)

    scope_color = (
        (0, 0, 255) if firing else (0, 255, 0)
    )

    # Scope boundary.
    cv2.circle(
        frame,
        (center_x, center_y),
        radius,
        scope_color,
        2
    )

    # Crosshairs.
    cv2.line(
        frame,
        (center_x - radius, center_y),
        (center_x + radius, center_y),
        (255, 255, 255),
        1
    )

    cv2.line(
        frame,
        (center_x, center_y - radius),
        (center_x, center_y + radius),
        (255, 255, 255),
        1
    )

    # Center aiming dot.
    cv2.circle(
        frame,
        (center_x, center_y),
        5 if firing else 3,
        scope_color,
        -1
    )

    cv2.putText(
        frame,
        "LIVE SCOPE",
        (center_x - 65, max(25, center_y - radius - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        scope_color,
        2
    )

    if firing:
        cv2.putText(
            frame,
            "FIRE!",
            (center_x - 45, center_y + radius + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )


# -------------------- HUD -----------------------------------

def draw_hud(
    frame,
    game_state,
    hand_detected,
    finger_gun,
    eye_status,
    ear,
    action
):
    height, width = frame.shape[:2]

    # Draw the panel on a separate layer.
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (350, 225),
        (15, 15, 15),
        -1
    )

    cv2.addWeighted(
        overlay, 0.70, frame, 0.30, 0, frame
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
            f"HAND: {'DETECTED' if hand_detected else 'NOT FOUND'}",
            (0, 255, 0) if hand_detected else (0, 0, 255)
        ),
        (
            f"GESTURE: {'FINGER GUN' if finger_gun else 'NONE'}",
            (0, 255, 0) if finger_gun else (180, 180, 180)
        ),
        (
            f"EYES: {eye_status}",
            (0, 0, 255) if eye_status == "CLOSED"
            else (255, 255, 255)
        ),
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
            0.48,
            color,
            1,
            cv2.LINE_AA
        )

        y += 25

    cv2.putText(
        frame,
        "Q: QUIT",
        (width - 110, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


# -------------------- MAIN PROGRAM -------------------------

def main():
    global state
    global eye_state
    global blink_start_time
    global last_fire_time
    global last_fire_display_time

    # Check model files.
    for model_path in (HAND_MODEL_PATH, FACE_MODEL_PATH):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(HAND_MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(FACE_MODEL_PATH)
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
            "Could not open webcam. Check CAMERA_INDEX."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    previous_timestamp_ms = 0

    print("=" * 48)
    print("CV SNIPER - STAGE 3")
    print("Hand + face tracking + scope + blink-to-fire")
    print("Raise your hand to equip the weapon.")
    print("Make a finger-gun gesture to scope in.")
    print("Blink while scoped to trigger FIRE.")
    print("Press Q to quit.")
    print("=" * 48)

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
                    print("Could not read a webcam frame.")
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = time.monotonic_ns() // 1_000_000
                timestamp_ms = max(
                    timestamp_ms,
                    previous_timestamp_ms + 1
                )
                previous_timestamp_ms = timestamp_ms

                # Run both detectors.
                hand_result = hand_detector.detect_for_video(
                    mp_image, timestamp_ms
                )

                face_result = face_detector.detect_for_video(
                    mp_image, timestamp_ms
                )

                # ---------------- HAND DETECTION ----------------

                hand_detected = bool(hand_result.hand_landmarks)

                hand = None
                finger_gun = False
                hand_raised = False

                if hand_detected:
                    hand = hand_result.hand_landmarks[0]

                    finger_gun = detect_finger_gun(hand)
                    hand_raised = is_hand_raised(hand)

                # ---------------- GAME STATE --------------------

                if state == GameState.IDLE:
                    if hand_detected and hand_raised:
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

                # ---------------- BLINK DETECTION ----------------

                blink_event = False
                ear = 0.0

                if face_result.face_landmarks:
                    face = face_result.face_landmarks[0]

                    ear = get_eye_aspect_ratio(face)

                    if eye_state == EyeState.OPEN:
                        if ear < CLOSED_THRESHOLD:
                            eye_state = EyeState.CLOSED
                            blink_start_time = timestamp_ms

                    elif eye_state == EyeState.CLOSED:
                        if ear > OPEN_THRESHOLD:
                            blink_duration = (
                                timestamp_ms - blink_start_time
                            )

                            eye_state = EyeState.OPEN

                            if (
                                MIN_BLINK_DURATION_MS
                                <= blink_duration
                                <= MAX_BLINK_DURATION_MS
                            ):
                                blink_event = True

                # ---------------- FIRE EVENT --------------------

                if (
                    state == GameState.SCOPED
                    and blink_event
                    and timestamp_ms - last_fire_time
                    >= FIRE_COOLDOWN_MS
                ):
                    last_fire_time = timestamp_ms
                    last_fire_display_time = timestamp_ms

                    print(f"[{timestamp_ms}] FIRE EVENT")

                firing = (
                    timestamp_ms - last_fire_display_time
                    < FIRE_DISPLAY_MS
                )

                action = "FIRE" if firing else "READY"
                eye_status = eye_state.name

                # ---------------- RENDERING ORDER ----------------
                # 1. Scope background
                # 2. Hand skeleton
                # 3. Face and eye landmarks
                # 4. HUD on top

                if state == GameState.SCOPED:
                    draw_scope(frame, firing=firing)

                if hand is not None:
                    draw_hand_landmarks(frame, hand)

                if face_result.face_landmarks:
                    draw_face_landmarks(
                        frame,
                        face_result.face_landmarks[0]
                    )
                else:
                    cv2.putText(
                        frame,
                        "FACE TRACKING: NO FACE",
                        (frame.shape[1] - 290, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                draw_hud(
                    frame=frame,
                    game_state=state,
                    hand_detected=hand_detected,
                    finger_gun=finger_gun,
                    eye_status=eye_status,
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
