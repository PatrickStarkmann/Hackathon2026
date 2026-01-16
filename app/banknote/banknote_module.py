"""Banknote recognition placeholder."""

from __future__ import annotations

from app.common import Decision
from app.banknote.ocr_stub import OcrBanknoteStub
from app.banknote.tflite_stub import TfliteBanknoteStub


class BanknoteEngine:
    """Facade for banknote recognition backends."""

    def __init__(self) -> None:
        self._ocr = OcrBanknoteStub()
        self._tflite = TfliteBanknoteStub()

    def predict(self, frame) -> Decision:
        decision = self._tflite.predict(frame)
        if decision.text_to_say != "Not implemented":
            return decision
        return self._ocr.predict(frame)
