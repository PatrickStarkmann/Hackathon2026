"""Banknote recognition placeholder."""

from __future__ import annotations

from app.common import Decision
from app.banknote.ocr_stub import OcrBanknoteStub
from app.banknote.tflite_stub import TfliteBanknoteStub


class BanknoteEngine:
    """Facade for banknote recognition backends."""

    def __init__(self, backend: str = "tflite") -> None:
        self._backend = backend
        self._ocr = OcrBanknoteStub()
        self._tflite = TfliteBanknoteStub()

    def predict(self, frame) -> Decision:
        if self._backend == "ocr":
            decision = self._ocr.predict(frame)
            if decision.text_to_say:
                return decision
            return Decision(
                text_to_say="Banknoten-Erkennung nicht eingerichtet. Bitte TFLite-Modell oder OCR installieren.",
                debug_text=decision.debug_text,
                conf=0.0,
            )
        decision = self._tflite.predict(frame)
        if decision.text_to_say:
            return decision
        ocr_decision = self._ocr.predict(frame)
        if ocr_decision.text_to_say:
            return ocr_decision
        return Decision(
            text_to_say="Banknoten-Erkennung nicht eingerichtet. Lege assets/banknote.tflite ab oder installiere OCR.",
            debug_text=f"{decision.debug_text} | {ocr_decision.debug_text}",
            conf=0.0,
        )
