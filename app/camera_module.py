"""Camera capture module."""

from __future__ import annotations

import importlib.util
import logging
import time
from typing import Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

Frame = "np.ndarray" if TYPE_CHECKING else Any


class CameraStream:
    """Simple camera stream wrapper."""

    def __init__(self, camera_index: int = 0) -> None:
        self._logger = logging.getLogger(__name__)
        self._cv2 = None
        if importlib.util.find_spec("cv2") is None:
            self._logger.error("OpenCV not installed. Install requirements.txt to use camera.")
            self._cap = None
        else:
            import cv2

            self._cv2 = cv2
            self._cap = cv2.VideoCapture(camera_index)
        self._last_ts = time.time()
        self._fps = 0.0

    @property
    def fps(self) -> float:
        return self._fps

    def read(self) -> Optional[Frame]:
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        now = time.time()
        dt = now - self._last_ts
        if dt > 0:
            self._fps = 1.0 / dt
        self._last_ts = now
        return frame

    def release(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
