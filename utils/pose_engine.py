import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

class PoseEngine:
    def __init__(self):
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path="models/pose_landmarker.task"
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def process(self, frame):
        # Convert OpenCV BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ✅ Convert NumPy array → MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(mp_image)

        if result.pose_landmarks:
            return result.pose_landmarks[0]

        return None
