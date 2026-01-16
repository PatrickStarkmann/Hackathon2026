"""Main integration entry point."""

from __future__ import annotations

import logging
import os
from typing import List

import cv2

from app.banknote.banknote_module import BanknoteEngine
from app.camera_module import CameraStream
from app.common import Detection
from app.config import DEBUG_DRAW, WINDOW_NAME
from app.logic_module import DecisionEngine
from app.speech_module import SpeechEngine
from app.vision_module import VisionEngine
from app.voice.commands import key_to_mode
from app.voice.interaction_controller import InteractionController


def _draw_debug(frame, detections: List[Detection], fps: float, mode: str) -> None:
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

    if DEBUG_DRAW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    mode = "idle"
    try:
        while True:
            frame = camera.read()
            if frame is None:
                logging.warning("No frame received from camera.")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            detections = vision.detect(frame)
            if DEBUG_DRAW:
                _draw_debug(frame, detections, camera.fps, mode)
                cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            mode = key_to_mode(key)
            if mode == "idle":
                continue

            if mode == "banknote":
                decision = banknote.predict(frame)
            else:
                decision = logic.decide(mode, detections, frame.shape)

            if decision.text_to_say:
                spoken_text = decision.text_to_say
                if decision.label and mode in ("identify", "count", "price", "full"):
                    formatted = interaction.format_for_command(
                        mode,
                        gegenstand=decision.label,
                        anzahl=decision.count,
                        preis_cent=decision.price_cent,
                        position_text=decision.position_text,
                    )
                    if formatted:
                        spoken_text = formatted
                speech.speak(spoken_text)
                logging.info("Decision: %s", decision.debug_text)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
