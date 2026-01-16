"""Decision logic module."""
#test
from __future__ import annotations

import logging
from typing import List, Tuple

from app.common import Decision, Detection
from app.config import CONF_THRESHOLD, OBSTACLE_AREA_THRESHOLD, VOTE_MIN, VOTE_N
from app.logic.aggregator import VoteAggregator


class DecisionEngine:
    """Decides what to say based on detections and mode."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._label_votes = VoteAggregator(maxlen=VOTE_N)

    def decide(
        self, mode: str, detections: List[Detection], frame_shape: Tuple[int, int, int]
    ) -> Decision:
        if mode == "identify":
            return self._identify(detections, frame_shape)
        if mode == "count":
            return self._count(detections)
        if mode == "obstacle":
            return self._obstacle(detections, frame_shape)
        return Decision(text_to_say="", debug_text="Idle", conf=0.0)

    def _identify(self, detections: List[Detection], frame_shape: Tuple[int, int, int]) -> Decision:
        if not detections:
            return self._uncertain("Kein Objekt erkannt")
        top = max(detections, key=lambda det: det.conf)
        self._label_votes.add(top.label)
        stable_label = self._label_votes.majority(VOTE_MIN)
        if top.conf < CONF_THRESHOLD or stable_label is None:
            return self._uncertain("Unsicher, bitte näher ran oder mehr Licht")
        position = self._position(top.bbox, frame_shape)
        text = f"{top.label}, {top.conf:.2f}, {position}"
        debug = f"Identify {top.label} ({top.conf:.2f}) {position}"
        return Decision(text_to_say=text, debug_text=debug, conf=top.conf)

    def _count(self, detections: List[Detection]) -> Decision:
        if not detections:
            return self._uncertain("Keine Objekte gezählt")
        label_counts = {}
        for det in detections:
            label_counts[det.label] = label_counts.get(det.label, 0) + 1
        dominant_label = max(label_counts.items(), key=lambda item: item[1])[0]
        self._label_votes.add(dominant_label)
        stable_label = self._label_votes.majority(VOTE_MIN)
        if stable_label is None:
            return self._uncertain("Unsicher, bitte ruhiger halten")
        count = label_counts.get(stable_label, 0)
        text = f"{count} {stable_label}"
        debug = f"Count {stable_label}: {count}"
        return Decision(text_to_say=text, debug_text=debug, conf=1.0)

    def _obstacle(
        self, detections: List[Detection], frame_shape: Tuple[int, int, int]
    ) -> Decision:
        height, width = frame_shape[:2]
        frame_area = float(height * width)
        for det in detections:
            if det.conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = det.bbox
            bbox_area = float(max(0, x2 - x1) * max(0, y2 - y1))
            if bbox_area / frame_area < OBSTACLE_AREA_THRESHOLD:
                continue
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            if (width * 0.3) < center_x < (width * 0.7) and (height * 0.3) < center_y < (
                height * 0.7
            ):
                return Decision(
                    text_to_say="Hindernis nah",
                    debug_text="Obstacle detected",
                    conf=det.conf,
                )
        return Decision(text_to_say="Freier Weg", debug_text="No obstacle", conf=0.0)

    def _position(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> str:
        _, width = frame_shape[:2]
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2.0
        if center_x < width / 3:
            return "links"
        if center_x > 2 * width / 3:
            return "rechts"
        return "mitte"

    def _uncertain(self, reason: str) -> Decision:
        self._logger.debug("Uncertain: %s", reason)
        return Decision(text_to_say=reason, debug_text=reason, conf=0.0)
