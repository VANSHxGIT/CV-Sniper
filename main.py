import cv2
import mediapipe as mp

from pathlib import Path

from cv_controller.config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)


# --------------------------------------------------
# MediaPipe Tasks
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"


def main():

    # ----------------------------------------------
    # Check model
    # ----------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hand Landmarker model not found:\n{MODEL_PATH}\n\n"
            "Download hand_landmarker.task and place it inside the models folder."
        )

    # ----------------------------------------------
    # Camera
    # ----------------------------------------------

    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. "
            "Check CAMERA_INDEX in config.py."
        )

    # ----------------------------------------------
    # Hand Landmarker configuration
    # ----------------------------------------------

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),

        running_mode=VisionRunningMode.VIDEO,

        num_hands=2,

        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,

        min_hand_presence_confidence=MIN_DETECTION_CONFIDENCE,

        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    # ----------------------------------------------
    # Create detector
    # ----------------------------------------------

    with HandLandmarker.create_from_options(options) as detector:

        frame_timestamp_ms = 0

        while True:

            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror webcam
            frame = cv2.flip(frame, 1)

            # OpenCV BGR → MediaPipe RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Create MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamp must increase for VIDEO mode
            frame_timestamp_ms += 33

            # Run hand detection
            result = detector.detect_for_video(
                mp_image,
                frame_timestamp_ms
            )

            # --------------------------------------
            # Draw detected hands
            # --------------------------------------

            if result.hand_landmarks:

                for hand_index, landmarks in enumerate(
                    result.hand_landmarks
                ):

                    # Draw points
                    for landmark in landmarks:

                        x = int(
                            landmark.x * frame.shape[1]
                        )

                        y = int(
                            landmark.y * frame.shape[0]
                        )

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
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

                        (5, 9),
                        (9, 13),
                        (13, 17),
                    ]

                    for start, end in connections:

                        x1 = int(
                            landmarks[start].x * frame.shape[1]
                        )

                        y1 = int(
                            landmarks[start].y * frame.shape[0]
                        )

                        x2 = int(
                            landmarks[end].x * frame.shape[1]
                        )

                        y2 = int(
                            landmarks[end].y * frame.shape[0]
                        )

                        cv2.line(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                    # ----------------------------------
                    # Hand label
                    # ----------------------------------

                    if result.handedness:

                        handedness = result.handedness[
                            hand_index
                        ][0]

                        label = handedness.display_name

                        wrist = landmarks[0]

                        text_x = int(
                            wrist.x * frame.shape[1]
                        )

                        text_y = int(
                            wrist.y * frame.shape[0]
                        ) - 20

                        cv2.putText(
                            frame,
                            label,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

            # --------------------------------------
            # UI
            # --------------------------------------

            cv2.putText(
                frame,
                "CV SNIPER",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "HAND TRACKING: ACTIVE",
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press Q or ESC to quit",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1
            )

            cv2.imshow(
                "CV Sniper - Phase 1",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()