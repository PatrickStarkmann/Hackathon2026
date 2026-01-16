"""Vision module using YOLOv8."""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from app.common import Detection


class VisionEngine:
    """YOLOv8 detection engine with dummy fallback."""

    def __init__(self, weights_path: str) -> None:
        self._weights_path = weights_path
        self._logger = logging.getLogger(__name__)
        self._dummy_mode = not os.path.exists(weights_path)
        self._model = None
        if importlib.util.find_spec("ultralytics") is None:
            self._logger.error(
                "Ultralytics not installed. Install requirements.txt to use YOLO."
            )
            self._dummy_mode = True
        if self._dummy_mode:
            self._logger.warning(
                "YOLO weights not found at %s. Running in dummy mode.",
                weights_path,
            )
        else:
            from ultralytics import YOLO

            self._model = YOLO(weights_path)

    def detect(self, frame: Any) -> List[Detection]:
        if frame is None:
            return []
        if self._dummy_mode:
            height, width = frame.shape[:2]
            box_w = int(width * 0.3)
            box_h = int(height * 0.3)
            x1 = (width - box_w) // 2
            y1 = (height - box_h) // 2
            x2 = x1 + box_w
            y2 = y1 + box_h
            return [Detection(label="dummy_item", conf=0.5, bbox=(x1, y1, x2, y2))]
        results = self._model(frame, verbose=False)
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf.item())
                cls_idx = int(box.cls.item())
                label = self._model.names.get(cls_idx, str(cls_idx))
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    Detection(label=label, conf=conf, bbox=(x1, y1, x2, y2))
                )
        return detections
