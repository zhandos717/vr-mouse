import logging
import os
import signal
import sys
from enum import Enum

import cv2
import pyautogui

from face_mouse_controller import FaceMouseController
from face_tracker import FaceTracker
from hand_tracker import HandTracker
from mouse_controller import MouseController
from voice_controller import VoiceController

_shutdown = False


class ControlMode(Enum):
    HAND = "hand"
    FACE = "face"


def _signal_handler(signum, _frame):
    global _shutdown
    _shutdown = True


def configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "DEBUG").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.DEBUG),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _draw_hud(frame, mode: ControlMode, scroll_mode: bool = False) -> None:
    """Draw mode indicator overlay on the frame."""
    text = f"Mode: {mode.value}"
    if scroll_mode:
        text += " [SCROLL]"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "h=hand  f=face  ESC=quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def _handle_voice_commands(
    commands: list[str],
    mode: ControlMode,
    log: logging.Logger,
) -> ControlMode:
    """Process voice commands, return (possibly updated) mode."""
    for cmd in commands:
        if cmd == "hand_mode":
            log.info("Voice: switching to HAND mode")
            mode = ControlMode.HAND
        elif cmd == "face_mode":
            log.info("Voice: switching to FACE mode")
            mode = ControlMode.FACE
        elif cmd == "click":
            pyautogui.click(_pause=False)
            log.info("Voice: click")
        elif cmd == "right_click":
            pyautogui.rightClick(_pause=False)
            log.info("Voice: right click")
        elif cmd == "scroll_up":
            pyautogui.scroll(3, _pause=False)
            log.info("Voice: scroll up")
        elif cmd == "scroll_down":
            pyautogui.scroll(-3, _pause=False)
            log.info("Voice: scroll down")
    return mode


def main() -> None:
    configure_logging()
    log = logging.getLogger(__name__)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open camera")
        sys.exit(1)

    hand_tracker = HandTracker()
    hand_controller = MouseController()

    face_tracker: FaceTracker | None = None
    face_controller: FaceMouseController | None = None
    try:
        face_tracker = FaceTracker()
        face_controller = FaceMouseController()
        log.info("Face tracker initialised")
    except Exception as e:
        log.warning("Face tracker unavailable (missing model?): %s", e)

    voice_ctrl = VoiceController()
    voice_ctrl.start()

    mode = ControlMode.HAND
    log.info("Control started in %s mode. Press 'h'/'f' to switch, ESC to quit.", mode.value)

    # Diagnostic: check if pyautogui can move the cursor
    try:
        x, y = pyautogui.position()
        pyautogui.moveTo(x + 1, y + 1, _pause=False)
        pyautogui.moveTo(x, y, _pause=False)
        log.info("pyautogui moveTo works (Accessibility granted)")
    except Exception as e:
        log.error("pyautogui moveTo FAILED — grant Accessibility in System Settings > Privacy: %s", e)

    frame_count = 0
    try:
        while not _shutdown:
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to read frame, skipping")
                continue

            # Process voice commands
            commands = voice_ctrl.drain_commands()
            if commands:
                mode = _handle_voice_commands(commands, mode, log)

            frame_count += 1
            scroll_mode = False

            if mode == ControlMode.HAND:
                hands = hand_tracker.process(frame)
                hand_tracker.draw_landmarks(frame)
                if hands:
                    hand_controller.update(hands[0])
                elif frame_count % 60 == 0:
                    log.debug("No hands detected (frame %d)", frame_count)

            elif mode == ControlMode.FACE and face_tracker is not None:
                face_data = face_tracker.process(frame)
                face_tracker.draw_landmarks(frame)
                if face_data:
                    face_controller.update(face_data)
                    scroll_mode = face_controller._scroll_mode
                elif frame_count % 60 == 0:
                    log.debug("No face detected (frame %d)", frame_count)

            _draw_hud(frame, mode, scroll_mode)
            cv2.imshow("VR Mouse", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                log.info("Exit requested by key")
                break
            elif key == ord("h"):
                mode = ControlMode.HAND
                log.info("Switched to HAND mode")
            elif key == ord("f"):
                if face_tracker is not None:
                    mode = ControlMode.FACE
                    log.info("Switched to FACE mode")
                else:
                    log.warning("Face tracker not available")
    finally:
        voice_ctrl.stop()
        hand_tracker.close()
        if face_tracker is not None:
            face_tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        log.info("Cleanup complete, exiting")


if __name__ == "__main__":
    main()
