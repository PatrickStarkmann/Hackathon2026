"""Lightweight smoke tests for camera, vision, and TTS."""

from __future__ import annotations

import logging
import os

from app.camera_module import CameraStream
from app.speech_module import SpeechEngine
from app.vision_module import VisionEngine


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    camera = CameraStream()
    frame = camera.read()
    if frame is None:
        logging.error("Camera test failed: no frame.")
    else:
        logging.info("Camera test passed.")
    camera.release()

    weights_path = os.path.join("assets", "yolo_weights.pt")
    vision = VisionEngine(weights_path=weights_path)
    if os.path.exists(weights_path):
        logging.info("YOLO weights found. Running a dummy detect call.")
        _ = vision.detect(frame) if frame is not None else []
        logging.info("Vision test passed.")
    else:
        logging.warning("YOLO weights missing, dummy mode expected.")

    speech = SpeechEngine()
    speech.speak("Smoke test: Text to speech OK")
    logging.info("TTS test invoked.")


if __name__ == "__main__":
    main()
