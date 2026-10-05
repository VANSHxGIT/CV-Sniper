import cv2
import mediapipe as mp
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/hand_landmarker.task"


# ---------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------

def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


def finger_is_extended(landmarks, tip, pip):
    """
    Determines whether a finger is extended by comparing
    the fingertip distance from the wrist against the PIP joint.
    """

    wrist = landmarks[0]

    tip_distance = distance(landmarks[tip], wrist)
    pip_distance = distance(landmarks[pip], wrist)

    return tip_distance > pip_distance * 1.15


def thumb_is_extended(landmarks):
    """
    Thumb uses a different geometry because it moves sideways
    relative to the other fingers.
    """

    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    return distance(thumb_tip, wrist) > distance(thumb_ip, wrist) * 1.1


# ---------------------------------------------------------
# Finger-gun classifier
# ---------------------------------------------------------

def detect_finger_gun(hand_landmarks):

    index_extended = finger_is_extended(
        hand_landmarks,
        tip=8,
        pip=6
    )

    middle_extended = finger_is_extended(
        hand_landmarks,
        tip=12,
        pip=10
    )

    ring_extended = finger_is_extended(
        hand_landmarks,
        tip=16,
        pip=14
    )

    pinky_extended = finger_is_extended(
        hand_landmarks,
        tip=20,
        pip=18
    )

    thumb_extended = thumb_is_extended(hand_landmarks)

    finger_gun = (
        index_extended
        and thumb_extended
        and not middle_extended
        and not ring_extended
        and not pinky_extended
    )

    return finger_gun, {
        "THUMB": thumb_extended,
        "INDEX": index_extended,
        "MIDDLE": middle_extended,
        "RING": ring_extended,
        "PINKY": pinky_extended,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    BaseOptions = python.BaseOptions
    VisionRunningMode = vision.RunningMode

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    start_time = time.time()

    with vision.HandLandmarker.create_from_options(options) as detector:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("ERROR: Could not read webcam frame.")
                break

            # Mirror webcam like a normal selfie camera
            frame = cv2.flip(frame, 1)

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

            # -------------------------------------------------
            # Draw / classify hand
            # -------------------------------------------------

            status = "NO HAND"

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                finger_gun, fingers = detect_finger_gun(hand)

                # Draw landmarks
                for landmark in hand:

                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

                # Draw connections
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20),
                    (5, 9), (9, 13), (13, 17)
                ]

                for a, b in connections:

                    x1 = int(hand[a].x * frame.shape[1])
                    y1 = int(hand[a].y * frame.shape[0])

                    x2 = int(hand[b].x * frame.shape[1])
                    y2 = int(hand[b].y * frame.shape[0])

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                if finger_gun:
                    status = "FINGER GUN"
                else:
                    status = "HAND DETECTED"

                # Finger states
                y = 40

                for name, state in fingers.items():

                    text = f"{name}: {'EXTENDED' if state else 'CURLED'}"

                    cv2.putText(
                        frame,
                        text,
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    y += 30

            # -------------------------------------------------
            # Main status
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"STATUS: {status}",
                (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "CV Sniper - Gesture Test",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()