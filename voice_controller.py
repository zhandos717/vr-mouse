import json
import logging
import os
import queue
import threading

log = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "vosk-model-small-en-us-0.15"
)

# Commands we recognise and their canonical names
_COMMAND_MAP = {
    "click": "click",
    "left click": "click",
    "right click": "right_click",
    "scroll up": "scroll_up",
    "scroll down": "scroll_down",
    "hand mode": "hand_mode",
    "face mode": "face_mode",
}


class VoiceController:
    def __init__(self, model_path: str | None = None):
        self._model_path = model_path or _DEFAULT_MODEL_DIR
        self._commands: queue.Queue[str] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return

        if not os.path.isdir(self._model_path):
            log.warning("Vosk model not found at %s — voice control disabled", self._model_path)
            return

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        log.info("Voice controller started (model: %s)", self._model_path)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        log.info("Voice controller stopped")

    def drain_commands(self) -> list[str]:
        """Non-blocking: return all queued commands since last call."""
        cmds: list[str] = []
        while True:
            try:
                cmds.append(self._commands.get_nowait())
            except queue.Empty:
                break
        return cmds

    def _listen_loop(self) -> None:
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model
        except ImportError as e:
            log.error("Cannot start voice controller: %s", e)
            self._running = False
            return

        try:
            model = Model(self._model_path)
        except Exception as e:
            log.error("Failed to load vosk model: %s", e)
            self._running = False
            return

        samplerate = 16000
        recognizer = KaldiRecognizer(model, samplerate)

        audio_q: queue.Queue[bytes] = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                log.debug("sounddevice status: %s", status)
            audio_q.put(bytes(indata))

        try:
            with sd.RawInputStream(
                samplerate=samplerate,
                blocksize=4000,
                dtype="int16",
                channels=1,
                callback=audio_callback,
            ):
                log.info("Listening for voice commands...")
                while self._running:
                    try:
                        data = audio_q.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").strip().lower()
                        if text:
                            self._parse_command(text)
        except Exception as e:
            log.error("Voice input stream error: %s", e)
        finally:
            self._running = False

    def _parse_command(self, text: str) -> None:
        log.debug("Heard: '%s'", text)
        for phrase, cmd in _COMMAND_MAP.items():
            if phrase in text:
                log.info("Voice command: %s (from '%s')", cmd, text)
                self._commands.put(cmd)
                return
