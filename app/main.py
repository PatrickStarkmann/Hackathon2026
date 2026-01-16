"""Main integration entry point."""

from __future__ import annotations

import logging
import os
from typing import List

import cv2

from app.banknote.banknote_module import BanknoteEngine
from app.camera_module import CameraStream
from app.common import Decision, Detection
from app.config import DEBUG_DRAW, WINDOW_NAME
from app.logic_module import DecisionEngine
from app.price.price_module import PriceEngine
from app.speech_module import SpeechEngine
from app.vision_module import VisionEngine
from app.voice.commands import key_to_mode
from app.voice.interaction_controller import InteractionController


def _draw_debug(
    frame, detections: List[Detection], fps: float, mode: str, last_decision: Decision | None
) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.label} {det.conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | Mode: {mode}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )
    if last_decision is not None:
        cv2.putText(
            frame,
            last_decision.debug_text,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        if last_decision.text_to_say:
            cv2.putText(
                frame,
                f"say: {last_decision.text_to_say}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )


def _extract_roi(frame, detections: List[Detection]):
    if not detections:
        return frame
    h, w = frame.shape[:2]
    best = max(detections, key=lambda det: (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1]))
    x1, y1, x2, y2 = best.bbox
    dx = int((x2 - x1) * 0.1)
    dy = int((y2 - y1) * 0.1)
    x1 = max(x1 - dx, 0)
    y1 = max(y1 - dy, 0)
    x2 = min(x2 + dx, w)
    y2 = min(y2 + dy, h)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def _format_spoken_text(
    decision: Decision, mode: str, interaction: InteractionController
) -> str:
    if not decision.text_to_say:
        return ""
    if decision.label and mode in {"identify", "count", "price", "full", "banknote"}:
        formatted = interaction.format_for_command(
            mode,
            gegenstand=decision.label,
            anzahl=decision.count,
            preis_cent=decision.price_cent,
            position_text=decision.position_text,
        )
        if formatted:
            return formatted
    return decision.text_to_say


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    weights_path = os.path.join("assets", "yolo_weights.pt")
    camera = CameraStream()
    vision = VisionEngine(
        weights_path=weights_path, allowed_labels=["banana", "orange", "bottle"]
    )
    logic = DecisionEngine()
    speech = SpeechEngine()
    interaction = InteractionController(speech)
    banknote = BanknoteEngine()
    price = PriceEngine()

    if DEBUG_DRAW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    mode = "idle"
    active_mode: str | None = None
    last_decision: Decision | None = None
    last_spoken_text = ""
    try:
        while True:
            frame = camera.read()
            if frame is None:
                logging.warning("No frame received from camera.")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            detections = vision.detect(frame)
            display_mode = active_mode or mode
            if DEBUG_DRAW:
                _draw_debug(frame, detections, camera.fps, display_mode, last_decision)
                cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            mode = key_to_mode(key)
            if mode in {"banknote", "price"}:
                active_mode = mode
            elif mode != "idle":
                active_mode = None
                if mode == "banknote":
                    decision = banknote.predict(frame)
                elif mode == "price":
                    decision = price.predict(frame)
                else:
                    decision = logic.decide(mode, detections, frame.shape)
                last_decision = decision
                spoken_text = _format_spoken_text(decision, mode, interaction)
                if spoken_text and spoken_text != last_spoken_text and speech.can_speak():
                    speech.speak(spoken_text)
                    last_spoken_text = spoken_text
                    logging.info("Decision: %s", decision.debug_text)

            if active_mode == "banknote":
                roi = _extract_roi(frame, detections)
                decision = banknote.predict(roi)
                last_decision = decision
                spoken_text = _format_spoken_text(decision, "banknote", interaction)
                if spoken_text and spoken_text != last_spoken_text and speech.can_speak():
                    speech.speak(spoken_text)
                    last_spoken_text = spoken_text
                    logging.info("Decision: %s", decision.debug_text)
            elif active_mode == "price":
                roi = _extract_roi(frame, detections)
                decision = price.predict(roi)
                last_decision = decision
                spoken_text = _format_spoken_text(decision, "price", interaction)
                if spoken_text and spoken_text != last_spoken_text and speech.can_speak():
                    speech.speak(spoken_text)
                    last_spoken_text = spoken_text
                    logging.info("Decision: %s", decision.debug_text)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
