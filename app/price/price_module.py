"""Price recognition via optional OCR."""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.common import Decision
from app.config import PRICE_VOTE_MIN, PRICE_VOTE_N
from app.logic.aggregator import VoteAggregator


class PriceEngine:
    """OCR-based price recognizer with stabilization and graceful fallback."""

    def __init__(self, vote_n: int = PRICE_VOTE_N, vote_min: int = PRICE_VOTE_MIN) -> None:
        self._logger = logging.getLogger(__name__)
        self._votes = VoteAggregator(maxlen=vote_n)
        self._vote_min = vote_min
        self._reason: Optional[str] = None
        try:
            import pytesseract  # noqa: F401
        except Exception:
            self._reason = "pytesseract missing"

    def predict(self, frame) -> Decision:
        if self._reason is not None:
            return Decision(
                text_to_say="Preis nicht erkannt. Bitte Preisschild nah und ruhig halten.",
                debug_text=self._reason,
                conf=0.0,
            )

        try:
            import cv2
            import pytesseract

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
            blur = cv2.GaussianBlur(resized, (3, 3), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(
                thresh, config="--psm 6 -c tessedit_char_whitelist=0123456789,.-"
            )
            match = re.search(r"(\d+[\.,]\d{2}|\d+)", text)
            if not match:
                return Decision(
                    text_to_say="Preis nicht erkannt. Bitte Preisschild nah und ruhig halten.",
                    debug_text="price_no_match",
                    conf=0.0,
                )
            value = match.group(1).replace(".", ",")
            self._votes.add(value)
            stable = self._votes.majority(self._vote_min)
            if stable is None:
                return Decision(
                    text_to_say="Unsicher. Bitte Preisschild ruhiger halten.",
                    debug_text="price_unstable",
                    conf=0.0,
                )
            return Decision(text_to_say=f"Preis {stable} Euro.", debug_text=f"price {stable}", conf=1.0)
        except Exception as exc:
            self._logger.error("Price OCR failed: %s", exc)
            return Decision(
                text_to_say="Preis nicht erkannt. Bitte Preisschild nah und ruhig halten.",
                debug_text="price_error",
                conf=0.0,
            )
