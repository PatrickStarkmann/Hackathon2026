"""Speech output using offline TTS."""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import time
from typing import Optional

from app.config import SPEAK_COOLDOWN_S


class SpeechEngine:
    """Offline text-to-speech with cooldown."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._engine: Optional[object] = None
        self._fallback_say = False
        self._say_voice: Optional[str] = None
        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._select_german_voice()
        except Exception as exc:
            self._logger.error("pyttsx3 init failed: %s", exc)
            if platform.system() == "Darwin" and shutil.which("say") is not None:
                self._fallback_say = True
                self._say_voice = "Anna"
                self._logger.warning("Falling back to macOS 'say' command for TTS.")
        self._last_spoken = 0.0

    def can_speak(self) -> bool:
        if self._engine is None and not self._fallback_say:
            return False
        return (time.time() - self._last_spoken) >= SPEAK_COOLDOWN_S

    def speak(self, text: str) -> None:
        if not text:
            return
        if not self.can_speak():
            self._logger.debug("Skipping speech due to cooldown.")
            return
        self._logger.info("Speaking: %s", text)
        if self._engine is not None:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
                self._last_spoken = time.time()
                return
            except Exception as exc:
                self._logger.error("pyttsx3 speak failed: %s", exc)
        if self._fallback_say:
            try:
                if self._say_voice:
                    subprocess.run(["say", "-v", self._say_voice, text], check=False)
                else:
                    subprocess.run(["say", text], check=False)
                self._last_spoken = time.time()
            except Exception as exc:
                self._logger.error("macOS say failed: %s", exc)
        else:
            self._logger.warning("TTS unavailable: no backend available.")

    def _select_german_voice(self) -> None:
        if self._engine is None:
            return
        try:
            voices = self._engine.getProperty("voices")
        except Exception as exc:
            self._logger.error("Failed to list voices: %s", exc)
            return
        for voice in voices:
            if self._is_german_voice(voice):
                try:
                    self._engine.setProperty("voice", voice.id)
                    self._logger.info("Using German voice: %s", voice.id)
                    return
                except Exception as exc:
                    self._logger.error("Failed to set voice: %s", exc)
                    return
        self._logger.warning("No German voice found; using default voice.")

    @staticmethod
    def _is_german_voice(voice) -> bool:
        fields = [getattr(voice, "id", ""), getattr(voice, "name", "")]
        langs = getattr(voice, "languages", []) or []
        for lang in langs:
            if isinstance(lang, bytes):
                try:
                    fields.append(lang.decode("utf-8", errors="ignore"))
                except Exception:
                    continue
            else:
                fields.append(str(lang))
        haystack = " ".join(fields).lower()
        return "de" in haystack or "german" in haystack or "de_de" in haystack
