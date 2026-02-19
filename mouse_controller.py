import logging
import math

import pyautogui

log = logging.getLogger(__name__)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# MediaPipe landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

# Zone mapping: landmarks in [ZONE_MIN, ZONE_MAX] map to full screen
ZONE_MIN = 0.15
ZONE_MAX = 0.85

CLICK_DISTANCE = 0.045
SMOOTHING = 0.3


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _remap(value: float, lo: float, hi: float) -> float:
    """Remap value from [lo, hi] to [0, 1], clamped."""
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _fingers_folded(lm: dict[int, tuple[float, float, float]]) -> bool:
    """Check if index, middle, ring, pinky are folded (tips below MCPs in y)."""
    for tip, mcp in (
        (INDEX_TIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_MCP),
        (RING_TIP, RING_MCP),
        (PINKY_TIP, PINKY_MCP),
    ):
        if lm[tip][1] < lm[mcp][1]:  # y increases downward in normalised coords
            return False
    return True


class MouseController:
    def __init__(self):
        self._screen_w, self._screen_h = pyautogui.size()
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None
        self._left_pressed = False
        self._right_pressed = False
        self._prev_wrist_y: float | None = None

    def update(self, landmarks: dict[int, tuple[float, float, float]]) -> None:
        if not landmarks:
            return

        self._handle_cursor(landmarks)
        self._handle_left_click(landmarks)
        self._handle_right_click(landmarks)
        self._handle_scroll(landmarks)

    def _handle_cursor(self, lm: dict[int, tuple[float, float, float]]) -> None:
        tip = lm[INDEX_TIP]
        # Mirror X for natural control, remap from zone to full range
        raw_x = 1.0 - tip[0]
        raw_y = tip[1]

        nx = _remap(raw_x, ZONE_MIN, ZONE_MAX)
        ny = _remap(raw_y, ZONE_MIN, ZONE_MAX)

        target_x = nx * self._screen_w
        target_y = ny * self._screen_h

        if self._smooth_x is None:
            self._smooth_x = target_x
            self._smooth_y = target_y
        else:
            self._smooth_x += SMOOTHING * (target_x - self._smooth_x)
            self._smooth_y += SMOOTHING * (target_y - self._smooth_y)

        pyautogui.moveTo(int(self._smooth_x), int(self._smooth_y), _pause=False)

    def _handle_left_click(self, lm: dict[int, tuple[float, float, float]]) -> None:
        dist = _distance(lm[INDEX_TIP], lm[THUMB_TIP])
        if dist < CLICK_DISTANCE:
            if not self._left_pressed:
                log.info("LEFT CLICK (dist=%.4f)", dist)
                pyautogui.click(_pause=False)
                self._left_pressed = True
        else:
            self._left_pressed = False

    def _handle_right_click(self, lm: dict[int, tuple[float, float, float]]) -> None:
        dist = _distance(lm[MIDDLE_TIP], lm[THUMB_TIP])
        if dist < CLICK_DISTANCE:
            if not self._right_pressed:
                log.info("RIGHT CLICK (dist=%.4f)", dist)
                pyautogui.rightClick(_pause=False)
                self._right_pressed = True
        else:
            self._right_pressed = False

    def _handle_scroll(self, lm: dict[int, tuple[float, float, float]]) -> None:
        if not _fingers_folded(lm):
            self._prev_wrist_y = None
            return

        wrist_y = lm[WRIST][1]
        if self._prev_wrist_y is not None:
            delta = self._prev_wrist_y - wrist_y  # positive = hand moved up
            if abs(delta) > 0.01:
                clicks = int(delta * 40)
                if clicks != 0:
                    log.info("SCROLL %+d", clicks)
                    pyautogui.scroll(clicks, _pause=False)
        self._prev_wrist_y = wrist_y
