"""TFLite banknote classifier stub."""

from __future__ import annotations

from app.common import Decision


class TfliteBanknoteStub:
    """Placeholder for a future TFLite classifier."""

    def predict(self, frame) -> Decision:
        return Decision(text_to_say="Not implemented", debug_text="TFLite stub", conf=0.0)
