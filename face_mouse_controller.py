import logging
import time

import pyautogui

from face_tracker import FaceData

log = logging.getLogger(__name__)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# Head rotation range (degrees) mapped to full screen
YAW_RANGE = 30.0   # ±30° yaw covers full screen width
PITCH_RANGE = 20.0  # ±20° pitch covers full screen height

SMOOTHING = 0.25

# Blink detection thresholds
BLINK_THRESHOLD = 0.4
BLINK_DEBOUNCE = 0.4  # seconds between clicks

# Jaw open threshold for scroll toggle
JAW_THRESHOLD = 0.5
SCROLL_SENSITIVITY = 3


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


class FaceMouseController:
    def __init__(self):
        self._screen_w, self._screen_h = pyautogui.size()
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None
        self._last_left_click: float = 0.0
        self._last_right_click: float = 0.0
        self._left_was_closed = False
        self._right_was_closed = False
        self._scroll_mode = False
        self._jaw_was_open = False
        self._prev_pitch: float | None = None

    def update(self, face: FaceData) -> None:
        if not face.landmarks:
            return

        self._handle_cursor(face)
        self._handle_blinks(face)
        self._handle_jaw_scroll(face)

    def _handle_cursor(self, face: FaceData) -> None:
        # Map yaw/pitch to screen coordinates
        # Yaw: negative = looking left, positive = looking right
        # We mirror so looking left moves cursor left on screen
        nx = _clamp01(0.5 - face.yaw / (2 * YAW_RANGE))
        # Pitch: negative = looking up, positive = looking down
        ny = _clamp01(0.5 + face.pitch / (2 * PITCH_RANGE))

        target_x = nx * self._screen_w
        target_y = ny * self._screen_h

        if self._smooth_x is None:
            self._smooth_x = target_x
            self._smooth_y = target_y
        else:
            self._smooth_x += SMOOTHING * (target_x - self._smooth_x)
            self._smooth_y += SMOOTHING * (target_y - self._smooth_y)

        pyautogui.moveTo(int(self._smooth_x), int(self._smooth_y), _pause=False)

    def _handle_blinks(self, face: FaceData) -> None:
        now = time.monotonic()

        # Left eye blink → left click
        left_closed = face.blink_left > BLINK_THRESHOLD and face.blink_right < BLINK_THRESHOLD
        if left_closed and not self._left_was_closed:
            if now - self._last_left_click > BLINK_DEBOUNCE:
                log.info("LEFT CLICK (blink_left=%.2f)", face.blink_left)
                pyautogui.click(_pause=False)
                self._last_left_click = now
        self._left_was_closed = left_closed

        # Right eye blink → right click
        right_closed = face.blink_right > BLINK_THRESHOLD and face.blink_left < BLINK_THRESHOLD
        if right_closed and not self._right_was_closed:
            if now - self._last_right_click > BLINK_DEBOUNCE:
                log.info("RIGHT CLICK (blink_right=%.2f)", face.blink_right)
                pyautogui.rightClick(_pause=False)
                self._last_right_click = now
        self._right_was_closed = right_closed

    def _handle_jaw_scroll(self, face: FaceData) -> None:
        jaw_open = face.jaw_open > JAW_THRESHOLD

        # Toggle scroll mode on jaw open edge
        if jaw_open and not self._jaw_was_open:
            self._scroll_mode = not self._scroll_mode
            self._prev_pitch = None
            log.info("Scroll mode: %s", "ON" if self._scroll_mode else "OFF")
        self._jaw_was_open = jaw_open

        if not self._scroll_mode:
            self._prev_pitch = None
            return

        # In scroll mode, use pitch changes for scrolling
        if self._prev_pitch is not None:
            delta = self._prev_pitch - face.pitch  # positive = head tilted up = scroll up
            if abs(delta) > 0.5:
                clicks = int(delta * SCROLL_SENSITIVITY)
                if clicks != 0:
                    log.info("SCROLL %+d", clicks)
                    pyautogui.scroll(clicks, _pause=False)
        self._prev_pitch = face.pitch
