import logging
import math
import os
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

log = logging.getLogger(__name__)

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


@dataclass
class FaceData:
    landmarks: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    blink_left: float = 0.0
    blink_right: float = 0.0
    jaw_open: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0


def _yaw_pitch_from_matrix(matrix: np.ndarray) -> tuple[float, float]:
    """Extract yaw and pitch (degrees) from a 4x4 facial transformation matrix."""
    # Rotation sub-matrix
    r = matrix[:3, :3]
    # Yaw = rotation around Y axis
    yaw = math.degrees(math.atan2(r[0, 2], r[2, 2]))
    # Pitch = rotation around X axis
    pitch = math.degrees(math.asin(-max(-1.0, min(1.0, r[1, 2]))))
    return yaw, pitch


class FaceTracker:
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._last_result = None
        self._timestamp_ms = 0

    def process(self, frame: np.ndarray) -> FaceData | None:
        """Process a BGR frame and return FaceData or None if no face detected."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33
        self._last_result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not self._last_result.face_landmarks:
            return None

        face_landmarks = self._last_result.face_landmarks[0]
        lm_dict: dict[int, tuple[float, float, float]] = {}
        for idx, lm in enumerate(face_landmarks):
            lm_dict[idx] = (lm.x, lm.y, lm.z)

        data = FaceData(landmarks=lm_dict)

        # Extract blendshapes
        if self._last_result.face_blendshapes:
            blendshapes = self._last_result.face_blendshapes[0]
            bs_map = {bs.category_name: bs.score for bs in blendshapes}
            data.blink_left = bs_map.get("eyeBlinkLeft", 0.0)
            data.blink_right = bs_map.get("eyeBlinkRight", 0.0)
            data.jaw_open = bs_map.get("jawOpen", 0.0)

        # Extract yaw/pitch from transformation matrix
        if self._last_result.facial_transformation_matrixes:
            matrix = self._last_result.facial_transformation_matrixes[0]
            mat = np.array(matrix).reshape(4, 4) if not isinstance(matrix, np.ndarray) else matrix
            data.yaw, data.pitch = _yaw_pitch_from_matrix(mat)

        log.debug(
            "Face: yaw=%.1f pitch=%.1f blinkL=%.2f blinkR=%.2f jaw=%.2f",
            data.yaw, data.pitch, data.blink_left, data.blink_right, data.jaw_open,
        )
        return data

    def draw_landmarks(self, frame: np.ndarray) -> None:
        """Draw face contour landmarks on the frame."""
        if not self._last_result or not self._last_result.face_landmarks:
            return

        h, w = frame.shape[:2]
        face_landmarks = self._last_result.face_landmarks[0]

        # Face oval indices (MediaPipe face mesh)
        FACE_OVAL = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
        ]
        # Left eye indices
        LEFT_EYE = [362, 385, 387, 263, 373, 380, 362]
        # Right eye indices
        RIGHT_EYE = [33, 160, 158, 133, 153, 144, 33]
        # Lips outer
        LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]

        for indices, color in [
            (FACE_OVAL, (200, 200, 200)),
            (LEFT_EYE, (0, 255, 255)),
            (RIGHT_EYE, (0, 255, 255)),
            (LIPS, (0, 128, 255)),
        ]:
            pts = []
            for i in indices:
                if i < len(face_landmarks):
                    lm = face_landmarks[i]
                    pts.append((int(lm.x * w), int(lm.y * h)))
            for j in range(len(pts) - 1):
                cv2.line(frame, pts[j], pts[j + 1], color, 1)

    def close(self):
        self._landmarker.close()
