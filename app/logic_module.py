"""Decision logic module."""

from __future__ import annotations

import logging
import time
from typing import List, Tuple, Dict

from app.common import Decision, Detection
from app.config import CONF_THRESHOLD, OBSTACLE_AREA_THRESHOLD, VOTE_MIN, VOTE_N
from app.logic.aggregator import VoteAggregator


class DecisionEngine:
    """Decides what to say based on detections and mode."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

        # getrennte Votes pro Modus, um Interferenzen zu vermeiden
        self._label_votes_identify = VoteAggregator(maxlen=VOTE_N)
        self._label_votes_count = VoteAggregator(maxlen=VOTE_N)

        # einfache Konfidenz-Glättung für Identify
        self._conf_history_identify: List[float] = []
        self._conf_maxlen_identify: int = 5

        # Obstacle: Cooldown, damit nicht ständig gesprochen wird
        self._last_obstacle_utter_time: float = 0.0
        self._obstacle_cooldown_seconds: float = 1.5

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

    # -----------------------
    # Identify-Modus
    # -----------------------

    def _identify(
        self, detections: List[Detection], frame_shape: Tuple[int, int, int]
    ) -> Decision:
        if not detections:
            return self._uncertain("Kein Produkt im Bild")

        # nach Konfidenz sortieren
        detections_sorted = sorted(detections, key=lambda det: det.conf, reverse=True)
        top = detections_sorted[0]

        # Mehrdeutigkeit: wenn das zweitbeste Objekt fast gleich sicher ist
        if len(detections_sorted) > 1:
            second = detections_sorted[1]
            conf_diff = top.conf - second.conf
            if conf_diff < 0.05 and top.label != second.label:
                return self._uncertain(
                    "Mehrere Produkte erkannt, bitte nur ein Produkt ins Bild halten"
                )

        # Konfidenz glätten
        smoothed_conf = self._smoothed_conf_identify(top.conf)

        # Hysterese um den Threshold herum
        # leicht strengere Grenze für "sicher"
        if smoothed_conf < (CONF_THRESHOLD - 0.05):
            return self._uncertain("Unsicher, Kamera näher an das Produkt halten")

        # Voting für stabilen Label über mehrere Frames
        self._label_votes_identify.add(top.label)
        stable_label = self._label_votes_identify.majority(VOTE_MIN)

        if stable_label is None:
            return self._uncertain("Unsicher, bitte Kamera ruhiger halten")

        position = self._position(top.bbox, frame_shape)
        pos_text = self._spoken_position(position)

        # Konfidenz nicht ansagen, nur nutzen
        text = f"{stable_label} {pos_text}"
        debug = f"Identify {stable_label} (raw={top.label}, conf={top.conf:.2f}, smooth={smoothed_conf:.2f}) {position}"
        return Decision(text_to_say=text, debug_text=debug, conf=smoothed_conf)

    # -----------------------
    # Count-Modus
    # -----------------------

    def _count(self, detections: List[Detection]) -> Decision:
        if not detections:
            return self._uncertain("Keine Produkte im Bild")

        # einfache Duplikat-Reduktion bei stark überlappenden Boxen
        filtered: List[Detection] = []
        for det in sorted(detections, key=lambda d: d.conf, reverse=True):
            is_duplicate = any(
                self._has_significant_overlap(det.bbox, existing.bbox)
                for existing in filtered
            )
            if not is_duplicate and det.conf >= CONF_THRESHOLD:
                filtered.append(det)

        if not filtered:
            return self._uncertain("Unsicher beim Zählen, bitte Kamera näher an die Produkte halten")

        label_counts: Dict[str, int] = {}
        for det in filtered:
            label_counts[det.label] = label_counts.get(det.label, 0) + 1

        # dominantes Label wählen (das am häufigsten vorkommt)
        dominant_label = max(label_counts.items(), key=lambda item: item[1])[0]

        # Voting für stabilen Typ        # Voting fuer stabilen Typ
        self._label_votes_count.add(dominant_label)
        stable_label = self._label_votes_count.majority(VOTE_MIN)

        # In Count-Modus nicht komplett blockieren, wenn noch nicht genug Votes da sind
        # oder das stabile Label im aktuellen Frame fehlt.
        chosen_label = (
            stable_label if stable_label in label_counts else dominant_label
        )

        count = label_counts.get(chosen_label, 0)
        if count == 0:
            return self._uncertain("Unsicher beim Zaehlen, bitte leicht die Perspektive aendern")

        text = f"{count} {chosen_label}"
        debug = (
            f"Count {chosen_label}: {count} "
            f"(filtered={len(filtered)}, raw={len(detections)}, "
            f"stable={stable_label})"
        )
        conf = 1.0 if stable_label is not None else 0.6
        return Decision(text_to_say=text, debug_text=debug, conf=conf)

    # -----------------------
    # Obstacle-Modus
    # -----------------------

    # def _obstacle(
    #     self, detections: List[Detection], frame_shape: Tuple[int, int, int]
    # ) -> Decision:
    #     height, width = frame_shape[:2]
    #     frame_area = float(height * width)

    #     best_det: Detection | None = None
    #     best_score: float = 0.0

    #     # wähle relevantes Hindernis (groß + sicher)
    #     for det in detections:
    #         if det.conf < CONF_THRESHOLD:
    #             continue
    #         x1, y1, x2, y2 = det.bbox
    #         bbox_area = float(max(0, x2 - x1) * max(0, y2 - y1))
    #         area_ratio = bbox_area / frame_area
    #         if area_ratio < OBSTACLE_AREA_THRESHOLD:
    #             continue

    #         # Score: Fläche * Konfidenz
    #         score = area_ratio * det.conf
    #         if score > best_score:
    #             best_score = score
    #             best_det = det

    #     now = time.time()

    #     if best_det is None:
    #         # Weg nur gelegentlich melden, nicht spammen
    #         if now - self._last_obstacle_utter_time < self._obstacle_cooldown_seconds:
    #             return Decision(text_to_say="", debug_text="No obstacle (cooldown)", conf=0.0)
    #         self._last_obstacle_utter_time = now
    #         return Decision(text_to_say="Freier Weg", debug_text="No obstacle", conf=0.0)

    #     # Richtung + grobe Entfernung
    #     x1, y1, x2, y2 = best_det.bbox
    #     bbox_area = float(max(0, x2 - x1) * max(0, y2 - y1))
    #     area_ratio = bbox_area / frame_area

    #     position = self._position(best_det.bbox, frame_shape)
    #     pos_text = self._spoken_position(position)

    #     if area_ratio > 0.25:
    #         dist_text = "sehr nah"
    #     elif area_ratio > 0.1:
    #         dist_text = "in deiner Nähe"
    #     else:
    #         dist_text = "weiter vorne"

    #     # Cooldown, damit „Hindernis nah“ nicht in jedem Frame kommt
    #     if now - self._last_obstacle_utter_time < self._obstacle_cooldown_seconds:
    #         return Decision(
    #             text_to_say="",
    #             debug_text=f"Obstacle suppressed by cooldown ({dist_text} {position})",
    #             conf=best_det.conf,
    #         )

    #     self._last_obstacle_utter_time = now
    #     text = f"Hindernis {dist_text} {pos_text}"
    #     debug = f"Obstacle {dist_text} {position}, area_ratio={area_ratio:.3f}"
    #     return Decision(text_to_say=text, debug_text=debug, conf=best_det.conf)

    # -----------------------
    # Hilfsfunktionen
    # -----------------------

    def _position(
        self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]
    ) -> str:
        _, width = frame_shape[:2]
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2.0
        if center_x < width / 3:
            return "links"
        if center_x > 2 * width / 3:
            return "rechts"
        return "mitte"

    def _spoken_position(self, position: str) -> str:
        """Mapping der internen Positionslabels zu gesprochenem Text."""
        if position == "links":
            return "links vor dir"
        if position == "rechts":
            return "rechts vor dir"
        return "direkt vor dir"

    def _uncertain(self, reason: str) -> Decision:
        self._logger.debug("Uncertain: %s", reason)
        return Decision(text_to_say=reason, debug_text=reason, conf=0.0)

    def _smoothed_conf_identify(self, new_conf: float) -> float:
        """Gleitender Durchschnitt für Identify-Konfidenz."""
        self._conf_history_identify.append(new_conf)
        if len(self._conf_history_identify) > self._conf_maxlen_identify:
            self._conf_history_identify.pop(0)
        return sum(self._conf_history_identify) / len(self._conf_history_identify)

    def _has_significant_overlap(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
        threshold: float = 0.5,
    ) -> bool:
        """Einfache IoU-Berechnung, um starke Überlappung zu erkennen."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = float(inter_w * inter_h)

        if inter_area <= 0:
            return False

        area1 = float(max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1]))
        area2 = float(max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1]))
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return False

        iou = inter_area / union_area
        return iou > threshold
