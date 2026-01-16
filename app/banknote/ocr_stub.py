"""OCR banknote recognizer stub."""

from __future__ import annotations

from app.common import Decision


class OcrBanknoteStub:
    """Placeholder for a future OCR-based classifier."""

    def predict(self, frame) -> Decision:
        return Decision(text_to_say="Not implemented", debug_text="OCR stub", conf=0.0)
