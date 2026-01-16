"""Speech output using offline TTS."""

from __future__ import annotations

import importlib.util
import logging
import time

from app.config import SPEAK_COOLDOWN_S


class SpeechEngine:
    """Offline text-to-speech with cooldown."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._engine = None
        if importlib.util.find_spec("pyttsx3") is None:
            self._logger.error("pyttsx3 not installed. Install requirements.txt for TTS.")
        else:
            import pyttsx3

            self._engine = pyttsx3.init()
        self._last_spoken = 0.0

    def can_speak(self) -> bool:
        if self._engine is None:
            return False
        return (time.time() - self._last_spoken) >= SPEAK_COOLDOWN_S

    def speak(self, text: str) -> None:
        if not text:
            return
        if self._engine is None:
            self._logger.warning("TTS unavailable: missing pyttsx3 dependency.")
            return
        if not self.can_speak():
            self._logger.debug("Skipping speech due to cooldown.")
            return
        self._logger.info("Speaking: %s", text)
        self._engine.say(text)
        self._engine.runAndWait()
        self._last_spoken = time.time()
