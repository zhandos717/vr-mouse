import logging
import os

import cv2
import mediapipe as mp
import numpy as np

log = logging.getLogger(__name__)

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
draw_landmarks = mp.tasks.vision.drawing_utils.draw_landmarks
DrawingSpec = mp.tasks.vision.drawing_utils.DrawingSpec


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_result = None
        self._timestamp_ms = 0

    def process(self, frame: np.ndarray) -> list[dict[int, tuple[float, float, float]]]:
        """Process a BGR frame and return normalised landmarks for each detected hand."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33  # ~30 fps
        self._last_result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        hands: list[dict[int, tuple[float, float, float]]] = []
        if not self._last_result.hand_landmarks:
            return hands

        for hand_landmarks in self._last_result.hand_landmarks:
            lm_dict: dict[int, tuple[float, float, float]] = {}
            for idx, lm in enumerate(hand_landmarks):
                lm_dict[idx] = (lm.x, lm.y, lm.z)
            hands.append(lm_dict)

        log.debug("Detected %d hand(s)", len(hands))
        return hands

    def draw_landmarks(self, frame: np.ndarray) -> None:
        """Draw landmarks on the frame in-place."""
        if not self._last_result or not self._last_result.hand_landmarks:
            return

        h, w = frame.shape[:2]
        for hand_landmarks in self._last_result.hand_landmarks:
            connections = HandLandmarksConnections.HAND_CONNECTIONS
            # Draw connections
            for conn in connections:
                start = hand_landmarks[conn.start]
                end = hand_landmarks[conn.end]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw points
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    def close(self):
        self._landmarker.close()
