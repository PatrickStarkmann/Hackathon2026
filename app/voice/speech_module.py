# speech_module.py
"""
Offline TTS engine with queue, cooldown, and repeat.

Design goals:
- Offline only (pyttsx3)
- No overlapping speech (single worker thread + queue)
- Cooldown to prevent audio spam (default: 2s)
- Repeat last utterance
- Stability > fancy features
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import pyttsx3


@dataclass(frozen=True)
class SpeechConfig:
    cooldown_seconds: float = 2.0   # max one speech every X seconds
    queue_maxsize: int = 30         # prevent unbounded memory growth
    rate: Optional[int] = None      # e.g. 170; None = keep default
    volume: Optional[float] = None  # 0.0..1.0; None = keep default
    voice_name_contains: Optional[str] = None  # pick a voice containing this substring


class SpeechEngine:
    """
    Public interface expected by the project:

    class SpeechEngine:
        def speak(self, text: str) -> None
        def can_speak(self) -> bool
    """

    def __init__(self, config: SpeechConfig | None = None) -> None:
        self.config = config or SpeechConfig()

        self._q: queue.Queue[Tuple[str, float]] = queue.Queue(maxsize=self.config.queue_maxsize)
        self._shutdown = threading.Event()
        self._engine_lock = threading.Lock()

        self._last_spoken: Optional[str] = None
        self._last_spoken_ts: float = 0.0

        # Start TTS engine and worker thread
        self._engine = pyttsx3.init()
        self._apply_config()

        self._worker = threading.Thread(target=self._run_worker, name="SpeechWorker", daemon=True)
        self._worker.start()

    # ---------------------------
    # Public API
    # ---------------------------

    def can_speak(self) -> bool:
        """Returns True if cooldown passed and engine is running."""
        if self._shutdown.is_set():
            return False
        return (time.monotonic() - self._last_spoken_ts) >= self.config.cooldown_seconds

    def speak(self, text: str) -> None:
        """
        Enqueue text to be spoken if it passes basic checks.
        Non-blocking: returns immediately.
        """
        text = (text or "").strip()
        if not text or self._shutdown.is_set():
            return

        # Cooldown check happens here to prevent queue spam.
        if not self.can_speak():
            return

        # Put into queue (drop if queue is full -> stability > completeness)
        try:
            self._q.put_nowait((text, time.monotonic()))
        except queue.Full:
            # Failure mode protection: don't block main loop
            return

    def repeat_last(self) -> None:
        """Repeat the last successfully spoken text (respects cooldown)."""
        if self._last_spoken:
            self.speak(self._last_spoken)

    def stop(self) -> None:
        """
        Immediately stop current speech if possible.
        (Does not clear queue by default.)
        """
        if self._shutdown.is_set():
            return
        with self._engine_lock:
            try:
                self._engine.stop()
            except Exception:
                pass

    def clear_queue(self) -> None:
        """Drop all queued utterances."""
        try:
            while True:
                self._q.get_nowait()
                self._q.task_done()
        except queue.Empty:
            return

    def shutdown(self) -> None:
        """Stop worker and release resources."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self.stop()
        # Unblock worker if it's waiting
        try:
            self._q.put_nowait(("", time.monotonic()))
        except queue.Full:
            pass

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _apply_config(self) -> None:
        with self._engine_lock:
            if self.config.rate is not None:
                try:
                    self._engine.setProperty("rate", int(self.config.rate))
                except Exception:
                    pass

            if self.config.volume is not None:
                try:
                    v = float(self.config.volume)
                    self._engine.setProperty("volume", max(0.0, min(1.0, v)))
                except Exception:
                    pass

            # Optional: pick voice by substring match (platform-dependent)
            if self.config.voice_name_contains:
                want = self.config.voice_name_contains.lower()
                try:
                    voices = self._engine.getProperty("voices") or []
                    for voice in voices:
                        name = (getattr(voice, "name", "") or "").lower()
                        vid = getattr(voice, "id", None)
                        if want in name and vid:
                            self._engine.setProperty("voice", vid)
                            break
                except Exception:
                    pass

    def _run_worker(self) -> None:
        """
        Single worker thread:
        - Takes queued texts
        - Speaks them sequentially via pyttsx3
        - Updates last_spoken + timestamp only when actually spoken
        """
        while not self._shutdown.is_set():
            try:
                text, enq_ts = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                # Ignore dummy wake item
                if not text.strip():
                    continue

                # Ensure cooldown is still satisfied at speak-time
                # (e.g. multiple producers might enqueue quickly)
                now = time.monotonic()
                remaining = self.config.cooldown_seconds - (now - self._last_spoken_ts)
                if remaining > 0:
                    # Wait a bit, but still allow shutdown
                    end = now + remaining
                    while time.monotonic() < end and not self._shutdown.is_set():
                        time.sleep(0.02)

                if self._shutdown.is_set():
                    break

                # Speak (blocking) but safely isolated in this worker thread
                with self._engine_lock:
                    self._engine.say(text)
                    self._engine.runAndWait()

                # Mark as last spoken only after success
                self._last_spoken = text
                self._last_spoken_ts = time.monotonic()

            except Exception:
                # Don't crash the worker in a demo.
                # If TTS fails, we just skip this utterance.
                pass
            finally:
                self._q.task_done()
