"""OCR banknote recognizer (optional)."""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.common import Decision
from app.config import BANKNOTE_VOTE_MIN, BANKNOTE_VOTE_N
from app.logic.aggregator import VoteAggregator


class OcrBanknoteStub:
    """OCR-based fallback for banknote recognition."""

    def __init__(self, vote_n: int = BANKNOTE_VOTE_N, vote_min: int = BANKNOTE_VOTE_MIN) -> None:
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
            return Decision(text_to_say="", debug_text=f"ocr_unavailable: {self._reason}", conf=0.0)

        try:
            import cv2
            import pytesseract

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(
                thresh, config="--psm 6 -c tessedit_char_whitelist=0123456789"
            )
            match = re.search(r"(5|10|20|50|100)", text)
            if not match:
                return Decision(
                    text_to_say="Unsicher. Bitte Schein flach halten und nah an die Kamera.",
                    debug_text="ocr_no_match",
                    conf=0.0,
                )
            label = match.group(1)
            self._votes.add(label)
            stable = self._votes.majority(self._vote_min)
            if stable is None:
                return Decision(
                    text_to_say="Unsicher. Bitte Schein flach halten und nah an die Kamera.",
                    debug_text="ocr_unstable",
                    conf=0.0,
                )
            return Decision(text_to_say=f"{stable} Euro.", debug_text=f"ocr {stable}", conf=1.0)
        except Exception as exc:
            self._logger.error("OCR failed: %s", exc)
            return Decision(text_to_say="", debug_text="ocr_unavailable: ocr_error", conf=0.0)
