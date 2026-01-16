"""Camera capture module."""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional, TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import numpy as np

Frame = "np.ndarray" if TYPE_CHECKING else Any


class CameraStream:
    """Simple camera stream wrapper."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        resize_to: tuple[int, int] | None = None,
        fps_smoothing: float = 0.9,
        backend: int | None = None,
        fallback_indices: Iterable[int] = (1, 2, 3),
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._cv2 = None
        self._cap = None
        self._width = width
        self._height = height
        self._resize_to = resize_to
        self._fps_smoothing = max(0.0, min(fps_smoothing, 0.99))
        self._last_ts = time.monotonic()
        self._fps = 0.0
        self._backend = backend

        try:
            import cv2
        except Exception as exc:
            self._logger.error("OpenCV not available: %s", exc)
            return

        self._cv2 = cv2
        if self._backend is None and sys.platform == "darwin":
            self._backend = cv2.CAP_AVFOUNDATION
        self._cap = self._open_camera(camera_index, fallback_indices)
        if self._cap is None:
            self._logger.error("No camera found. Try a different index or check permissions.")

    @property
    def fps(self) -> float:
        return self._fps

    def read(self) -> Optional[Frame]:
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        if self._resize_to is not None and self._cv2 is not None:
            frame = self._cv2.resize(frame, self._resize_to)
        self._update_fps()
        return frame

    def release(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

    def _open_camera(self, camera_index: int, fallback_indices: Iterable[int]) -> Optional[Any]:
        if self._cv2 is None:
            return None
        indices = [camera_index] + [idx for idx in fallback_indices if idx != camera_index]
        for idx in indices:
            cap = self._create_capture(idx)
            if cap is None:
                continue
            if cap.isOpened():
                self._configure_capture(cap)
                self._logger.info("Camera opened at index %s", idx)
                return cap
            cap.release()
        return None

    def _create_capture(self, index: int) -> Optional[Any]:
        if self._cv2 is None:
            return None
        try:
            if self._backend is not None:
                return self._cv2.VideoCapture(index, self._backend)
            return self._cv2.VideoCapture(index)
        except Exception as exc:
            self._logger.warning("Failed to open camera index %s: %s", index, exc)
            return None

    def _configure_capture(self, cap: Any) -> None:
        if self._width is not None:
            cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height is not None:
            cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self._height)

    def _update_fps(self) -> None:
        now = time.monotonic()
        dt = now - self._last_ts
        if dt > 0:
            inst = 1.0 / dt
            if self._fps == 0.0:
                self._fps = inst
            else:
                self._fps = (self._fps * self._fps_smoothing) + (inst * (1.0 - self._fps_smoothing))
        self._last_ts = now
