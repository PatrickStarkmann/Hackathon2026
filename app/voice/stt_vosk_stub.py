"""Offline speech-to-text using Vosk."""

from __future__ import annotations

from typing import Optional

import json
import logging
import os
import queue
import threading

import sounddevice as sd
import vosk


class VoskSttStub:
    """Offline Vosk-based STT listener."""

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        sample_rate: int = 16000,
        device: Optional[int] = None,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._sample_rate = sample_rate
        self._device = device
        self._model_path = model_path or os.environ.get("VOSK_MODEL_PATH", "assets/vosk-model")
        self._model: Optional[vosk.Model] = None

        if os.path.isdir(self._model_path):
            try:
                self._model = vosk.Model(self._model_path)
            except Exception as exc:
                self._logger.error("Vosk model init failed: %s", exc)
        else:
            self._logger.warning("Vosk model path not found: %s", self._model_path)

    def available(self) -> bool:
        return self._model is not None

    def listen(self) -> Optional[str]:
        """Blocking listen for a single utterance."""
        if not self._model:
            return None

        recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate)
        result_text: Optional[str] = None

        def _callback(indata: bytes, _frames: int, _time, _status) -> None:
            nonlocal result_text
            if _status:
                self._logger.debug("STT stream status: %s", _status)
            if recognizer.AcceptWaveform(bytes(indata)):
                result = json.loads(recognizer.Result())
                text = (result.get("text") or "").strip()
                if text:
                    result_text = text

        with sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=_callback,
            device=self._device,
        ):
            while result_text is None:
                sd.sleep(100)

        return result_text

    def listen_loop(self, out_queue: queue.Queue[str], stop_event: threading.Event) -> None:
        """Continuously push recognized utterances into a queue."""
        if not self._model:
            return

        recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate)

        def _callback(indata: bytes, _frames: int, _time, _status) -> None:
            if _status:
                self._logger.debug("STT stream status: %s", _status)
            if recognizer.AcceptWaveform(bytes(indata)):
                result = json.loads(recognizer.Result())
                text = (result.get("text") or "").strip()
                if text:
                    try:
                        out_queue.put_nowait(text)
                    except queue.Full:
                        return

        with sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=_callback,
            device=self._device,
        ):
            while not stop_event.is_set():
                sd.sleep(100)
