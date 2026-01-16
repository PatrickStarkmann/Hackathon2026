"""Vision module using YOLOv8."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, List, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import numpy as np

from app.common import Detection


class VisionEngine:
    """YOLOv8 detection engine with dummy fallback."""

    def __init__(self, weights_path: str, allowed_labels: List[str] | None = None) -> None:
        self._weights_path = Path(weights_path)
        self._logger = logging.getLogger(__name__)
        self._dummy_mode = not self._weights_path.exists()
        self._model = None
        self._allowed_labels = (
            {label.strip().lower() for label in allowed_labels}
            if allowed_labels
            else None
        )
        if importlib.util.find_spec("ultralytics") is None:
            self._logger.error(
                "Ultralytics not installed. Install requirements.txt to use YOLO."
            )
            self._dummy_mode = True
        if self._dummy_mode:
            self._logger.warning(
                "YOLO weights not found at %s. Running in dummy mode.",
                self._weights_path,
            )
        else:
            from ultralytics import YOLO

            try:
                self._model = YOLO(str(self._weights_path))
            except Exception as exc:
                self._logger.error("Failed to load YOLO weights: %s", exc)
                self._dummy_mode = True

    def detect(self, frame: Any) -> List[Detection]:
        if frame is None:
            return []
        if self._dummy_mode:
            return self._dummy_detection(frame)
        if self._model is None:
            return []
        results = self._model.predict(source=frame, verbose=False)
        detections: List[Detection] = []
        height, width = frame.shape[:2]
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            names = result.names or getattr(self._model, "names", {})
            for (x1, y1, x2, y2), conf, cls_idx in zip(xyxy, confs, classes):
                label = names.get(int(cls_idx), str(int(cls_idx)))
                if self._allowed_labels and label.lower() not in self._allowed_labels:
                    continue
                bbox = self._clip_bbox((x1, y1, x2, y2), width, height)
                detections.append(
                    Detection(label=label, conf=float(conf), bbox=bbox)
                )
        return detections

    def _dummy_detection(self, frame: "np.ndarray") -> List[Detection]:
        height, width = frame.shape[:2]
        box_w = int(width * 0.3)
        box_h = int(height * 0.3)
        x1 = (width - box_w) // 2
        y1 = (height - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h
        return [Detection(label="dummy_item", conf=0.5, bbox=(x1, y1, x2, y2))]

    @staticmethod
    def _clip_bbox(
        bbox: Tuple[float, float, float, float], width: int, height: int
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        x1_i = max(0, min(int(x1), width - 1))
        y1_i = max(0, min(int(y1), height - 1))
        x2_i = max(0, min(int(x2), width - 1))
        y2_i = max(0, min(int(y2), height - 1))
        return x1_i, y1_i, x2_i, y2_i
