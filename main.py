import cv2
import mediapipe as mp
import time
import math
from enum import Enum

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/hand_landmarker.task"


# =========================================================
# GAME STATES
# =========================================================

class GameState(Enum):
    IDLE = 0
    EQUIPPED = 1
    SCOPED = 2


# =========================================================
# GEOMETRY
# =========================================================

def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


def finger_is_extended(landmarks, tip, pip):

    wrist = landmarks[0]

    tip_distance = distance(
        landmarks[tip],
        wrist
    )

    pip_distance = distance(
        landmarks[pip],
        wrist
    )

    return tip_distance > pip_distance * 1.15


def thumb_is_extended(landmarks):

    wrist = landmarks[0]

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    return (
        distance(thumb_tip, wrist)
        >
        distance(thumb_ip, wrist) * 1.1
    )


# =========================================================
# GESTURE DETECTION
# =========================================================

def detect_finger_gun(hand):

    index = finger_is_extended(
        hand, 8, 6
    )

    middle = finger_is_extended(
        hand, 12, 10
    )

    ring = finger_is_extended(
        hand, 16, 14
    )

    pinky = finger_is_extended(
        hand, 20, 18
    )

    thumb = thumb_is_extended(hand)

    finger_gun = (
        index
        and thumb
        and not middle
        and not ring
        and not pinky
    )

    return finger_gun


# =========================================================
# HAND POSITION
# =========================================================

def get_hand_center(hand):

    x = sum(point.x for point in hand) / len(hand)
    y = sum(point.y for point in hand) / len(hand)

    return x, y


def is_hand_raised(hand):

    """
    Simple first version.

    The wrist/hand must be in the upper portion
    of the camera frame.
    """

    center_x, center_y = get_hand_center(hand)

    return center_y < 0.55


# =========================================================
# DRAW HAND
# =========================================================

def draw_hand(frame, hand):

    h, w = frame.shape[:2]

    # Landmarks
    for landmark in hand:

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )

    # Connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),

        (0, 5), (5, 6), (6, 7), (7, 8),

        (0, 9), (9, 10), (10, 11), (11, 12),

        (0, 13), (13, 14), (14, 15), (15, 16),

        (0, 17), (17, 18), (18, 19), (19, 20),

        (5, 9),
        (9, 13),
        (13, 17)
    ]

    for a, b in connections:

        x1 = int(hand[a].x * w)
        y1 = int(hand[a].y * h)

        x2 = int(hand[b].x * w)
        y2 = int(hand[b].y * h)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


# =========================================================
# HUD
# =========================================================

def draw_hud(frame, state, finger_gun):

    h, w = frame.shape[:2]

    # =====================================================
    # NORMAL HUD
    # =====================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (w, 70),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        "WEAPON: SNIPER RIFLE",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"STATE: {state.name}",
        (20, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    gesture_text = "FINGER GUN" if finger_gun else "NONE"

    cv2.putText(
        frame,
        f"GESTURE: {gesture_text}",
        (w - 280, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # =====================================================
    # SNIPER SCOPE
    # =====================================================

    if state == GameState.SCOPED:

        center_x = w // 2
        center_y = h // 2

        radius = int(min(w, h) * 0.38)

        # ---------------------------------------------
        # Darken entire screen
        # ---------------------------------------------

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (w, h),
            (0, 0, 0),
            -1
        )

        # Blend darkness over camera
        cv2.addWeighted(
            overlay,
            0.72,
            frame,
            0.28,
            0,
            frame
        )

        # ---------------------------------------------
        # Scope circle
        # ---------------------------------------------

        cv2.circle(
            frame,
            (center_x, center_y),
            radius,
            (220, 220, 220),
            5
        )

        # ---------------------------------------------
        # Crosshair
        # ---------------------------------------------

        crosshair_length = radius

        # Horizontal
        cv2.line(
            frame,
            (center_x - crosshair_length, center_y),
            (center_x + crosshair_length, center_y),
            (220, 220, 220),
            2
        )

        # Vertical
        cv2.line(
            frame,
            (center_x, center_y - crosshair_length),
            (center_x, center_y + crosshair_length),
            (220, 220, 220),
            2
        )

        # ---------------------------------------------
        # Center aiming point
        # ---------------------------------------------

        cv2.circle(
            frame,
            (center_x, center_y),
            7,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            frame,
            (center_x, center_y),
            13,
            (255, 255, 255),
            2
        )

        # ---------------------------------------------
        # Scope information
        # ---------------------------------------------

        cv2.putText(
            frame,
            "● LIVE SCOPE",
            (center_x - 90, center_y - radius + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            "TARGET ACQUISITION",
            (center_x - 110, center_y + radius - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            2
        )

# =========================================================
# MAIN
# =========================================================

def main():

    BaseOptions = python.BaseOptions
    RunningMode = vision.RunningMode

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=1
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")
        return

    state = GameState.IDLE

    start_time = time.time()

    with vision.HandLandmarker.create_from_options(
        options
    ) as detector:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(
                frame,
                1
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                (time.time() - start_time) * 1000
            )

            result = detector.detect_for_video(
                mp_image,
                timestamp_ms
            )

            hand_detected = False
            finger_gun = False
            hand_raised = False

            # =================================================
            # DETECTION
            # =================================================

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                hand_detected = True

                finger_gun = detect_finger_gun(
                    hand
                )

                hand_raised = is_hand_raised(
                    hand
                )

                draw_hand(
                    frame,
                    hand
                )

            # =================================================
            # STATE MACHINE
            # =================================================

            if state == GameState.IDLE:

                if hand_detected and hand_raised:

                    state = GameState.EQUIPPED

            elif state == GameState.EQUIPPED:

                # Lose the hand → holster
                if not hand_detected:

                    state = GameState.IDLE

                # Finger gun → scope
                elif finger_gun:

                    state = GameState.SCOPED

            elif state == GameState.SCOPED:

                # No hand → holster
                if not hand_detected:

                    state = GameState.IDLE

                # Stop finger gun → leave scope
                elif not finger_gun:

                    state = GameState.EQUIPPED

            # =================================================
            # HUD
            # =================================================

            draw_hud(
                frame,
                state,
                finger_gun
            )

            cv2.imshow(
                "CV Sniper",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()