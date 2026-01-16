# speech_module.py
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import pyttsx3


@dataclass(frozen=True)
class SpeechConfig:
    cooldown_seconds: float = 2.0
    queue_maxsize: int = 30
    rate: Optional[int] = None
    volume: Optional[float] = None
    voice_name_contains: Optional[str] = None


class SpeechEngine:
    def __init__(self, config: SpeechConfig | None = None) -> None:
        self.config = config or SpeechConfig()

        self._q: queue.Queue[Tuple[str, float]] = queue.Queue(maxsize=self.config.queue_maxsize)
        self._shutdown = threading.Event()
        self._stop_requested = threading.Event()

        self._last_spoken: Optional[str] = None
        self._last_spoken_ts: float = 0.0

        self._worker = threading.Thread(target=self._run_worker, name="SpeechWorker", daemon=True)
        self._worker.start()

    def can_speak(self) -> bool:
        if self._shutdown.is_set():
            return False
        return (time.monotonic() - self._last_spoken_ts) >= self.config.cooldown_seconds

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text or self._shutdown.is_set():
            return
        try:
            self._q.put_nowait((text, time.monotonic()))
        except queue.Full:
            return

    def repeat_last(self) -> None:
        if self._last_spoken:
            self.speak(self._last_spoken)

    def stop(self) -> None:
        """Request stop (worker will call engine.stop())."""
        if self._shutdown.is_set():
            return
        self._stop_requested.set()

    def clear_queue(self) -> None:
        try:
            while True:
                self._q.get_nowait()
                self._q.task_done()
        except queue.Empty:
            return

    def shutdown(self) -> None:
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self.stop()
        # Unblock worker
        try:
            self._q.put_nowait(("", time.monotonic()))
        except queue.Full:
            pass

    # -------- worker / TTS thread --------

    def _apply_config(self, engine: pyttsx3.Engine) -> None:
        if self.config.rate is not None:
            try:
                engine.setProperty("rate", int(self.config.rate))
            except Exception:
                pass

        if self.config.volume is not None:
            try:
                v = float(self.config.volume)
                engine.setProperty("volume", max(0.0, min(1.0, v)))
            except Exception:
                pass

        if self.config.voice_name_contains:
            want = self.config.voice_name_contains.lower()
            try:
                voices = engine.getProperty("voices") or []
                for voice in voices:
                    name = (getattr(voice, "name", "") or "").lower()
                    vid = getattr(voice, "id", None)
                    if want in name and vid:
                        engine.setProperty("voice", vid)
                        break
            except Exception:
                pass

    def _run_worker(self) -> None:
        # IMPORTANT: init pyttsx3 inside the worker thread (Windows/SAPI5 stability)
        engine = pyttsx3.init()
        self._apply_config(engine)

        while not self._shutdown.is_set():
            try:
                text, _ = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if not text.strip():
                    continue

                # cooldown enforced here (no spam, no dropping)
                now = time.monotonic()
                remaining = self.config.cooldown_seconds - (now - self._last_spoken_ts)
                if remaining > 0:
                    end = now + remaining
                    while time.monotonic() < end and not self._shutdown.is_set():
                        time.sleep(0.02)

                if self._shutdown.is_set():
                    break

                # stop requested?
                if self._stop_requested.is_set():
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    self._stop_requested.clear()

                engine.say(text)
                engine.runAndWait()

                self._last_spoken = text
                self._last_spoken_ts = time.monotonic()

            except Exception:
                # keep demo alive even if TTS glitches
                pass
            finally:
                self._q.task_done()

        try:
            engine.stop()
        except Exception:
            pass
