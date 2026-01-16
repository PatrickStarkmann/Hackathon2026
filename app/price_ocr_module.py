"""Offline OCR module for extracting prices from frames."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from typing import Any, List, Optional, Tuple

import cv2


@dataclass(frozen=True)
class OCRResult:
    text: str
    price: float | None
    conf: float
    debug_text: str


class PriceOCREngine:
    """Extracts prices from a camera frame using offline OCR."""

    _PRICE_MIN = 0.10
    _PRICE_MAX = 1000.0
    _CONF_THRESHOLD = 0.6

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._tesseract_available = self._check_tesseract()

    def extract_price(self, frame: Any) -> OCRResult:
        if frame is None:
            return OCRResult(
                text="Kein Preis erkennbar",
                price=None,
                conf=0.0,
                debug_text="No frame provided",
            )
        if not self._tesseract_available:
            return OCRResult(
                text="Texterkennung nicht verfügbar",
                price=None,
                conf=0.0,
                debug_text="pytesseract not installed",
            )
        try:
            ocr_texts: List[str] = []
            prices: List[float] = []
            for preprocessed in self._preprocess_variants(frame):
                ocr_text = self._run_ocr(preprocessed)
                if ocr_text:
                    ocr_texts.append(ocr_text)
                prices.extend(self._extract_prices(ocr_text))
            debug_text = " | ".join(
                f"OCR: {text} | Matches: {self._extract_prices(text)}"
                for text in ocr_texts
            )
            return self._build_result(debug_text, prices)
        except Exception as exc:
            self._logger.error("OCR failed: %s", exc)
            return OCRResult(
                text="Texterkennung nicht verfügbar",
                price=None,
                conf=0.0,
                debug_text=f"OCR error: {exc}",
            )

    def draw_debug(
        self, frame: Any, boxes: List[Tuple[int, int, int, int]]
    ) -> Any:
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
        return frame

    def _crop_for_price(self, frame: Any) -> Any:
        height, width = frame.shape[:2]
        top = int(height * 0.15)
        bottom = int(height * 0.9)
        left = int(width * 0.1)
        right = int(width * 0.9)
        if bottom <= top or right <= left:
            return frame
        return frame[top:bottom, left:right]

    def find_price_boxes(self, frame: Any) -> List[Tuple[int, int, int, int]]:
        if frame is None or not self._tesseract_available:
            return []
        try:
            import pytesseract

            preprocessed = self._preprocess_variants(frame)[0]
            config = (
                "--oem 3 --psm 6 "
                "-c tessedit_char_whitelist=0123456789€.,cent"
            )
            data = pytesseract.image_to_data(
                preprocessed, config=config, output_type=pytesseract.Output.DICT
            )
            boxes: List[Tuple[int, int, int, int]] = []
            decimal_pattern = re.compile(r"(?:€\s*)?\d{1,4}[.,]\d{2}(?:\s*€)?")
            cent_pattern = re.compile(r"\d{1,4}\s*cent")
            for i, word in enumerate(data.get("text", [])):
                token = (word or "").strip().lower()
                if not token:
                    continue
                if not (decimal_pattern.fullmatch(token) or cent_pattern.fullmatch(token)):
                    continue
                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])
                boxes.append((x, y, x + w, y + h))
            return boxes
        except Exception as exc:
            self._logger.debug("Price box detection failed: %s", exc)
            return []

    def _preprocess_variants(self, frame: Any) -> List[Any]:
        cropped = self._crop_for_price(frame)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        scale = 2.0
        resized = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
        cleaned_adaptive = cv2.morphologyEx(
            adaptive, cv2.MORPH_CLOSE, kernel, iterations=1
        )
        inv_otsu = cv2.bitwise_not(cleaned_otsu)
        inv_adaptive = cv2.bitwise_not(cleaned_adaptive)
        return [cleaned_otsu, cleaned_adaptive, inv_otsu, inv_adaptive]

    def _run_ocr(self, image: Any) -> str:
        import pytesseract

        config = (
            "--oem 3 --psm 6 "
            "-c tessedit_char_whitelist=0123456789€.,"
        )
        texts = [
            pytesseract.image_to_string(image, config=config),
            pytesseract.image_to_string(image, config="--oem 3 --psm 7"),
        ]
        return " ".join(texts).strip()

    def _extract_prices(self, text: str) -> List[float]:
        if not text:
            return []
        matches: List[float] = []
        normalized_text = self._normalize_text(text.lower())
        decimal_pattern = r"(?:€\s*)?(\d{1,4}[.,]\d{1,2})(?:\s*€)?"
        cent_pattern = r"(\d{1,4})\s*cent"
        for match in re.findall(decimal_pattern, normalized_text):
            price = self._normalize_price(match)
            if price is None:
                continue
            if self._PRICE_MIN <= price <= self._PRICE_MAX:
                matches.append(price)
        for match in re.findall(cent_pattern, normalized_text):
            try:
                price = float(match) / 100.0
            except ValueError:
                continue
            if self._PRICE_MIN <= price <= self._PRICE_MAX:
                matches.append(price)
        return matches

    def _normalize_price(self, match: str) -> Optional[float]:
        cleaned = match.replace(" ", "").replace("€", "")
        if not cleaned:
            return None
        normalized = cleaned.replace(",", ".")
        if normalized.count(".") == 1:
            left, right = normalized.split(".")
            if right.isdigit() and len(right) == 1:
                normalized = f"{left}.{right}0"
        try:
            value = float(normalized)
            if normalized.isdigit() and value >= 10:
                return value / 100.0
            return value
        except ValueError:
            return None

    def _build_result(self, debug_text: str, prices: List[float]) -> OCRResult:
        debug = debug_text or f"Matches: {prices}"
        if not prices:
            return OCRResult(
                text="Kein Preis erkennbar",
                price=None,
                conf=0.0,
                debug_text=debug,
            )
        rounded = [round(value, 2) for value in prices]
        counts: dict[float, int] = {}
        for value in rounded:
            counts[value] = counts.get(value, 0) + 1
        price = max(counts.items(), key=lambda item: item[1])[0]
        top_count = counts[price]
        if len(counts) == 1:
            conf = 0.9
        elif top_count >= 2:
            conf = 0.7
        else:
            conf = 0.4
        if conf >= self._CONF_THRESHOLD:
            text = f"Preis: {price:.2f} Euro"
            return OCRResult(text=text, price=price, conf=conf, debug_text=debug)
        return OCRResult(
            text="Kein Preis erkennbar",
            price=None,
            conf=conf,
            debug_text=debug,
        )

    def _check_tesseract(self) -> bool:
        try:
            import pytesseract
            from shutil import which

            tesseract_path = self._resolve_tesseract_path()
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            elif which("tesseract") is None:
                return False
            _ = pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def _resolve_tesseract_path(self) -> Optional[str]:
        env_path = os.environ.get("TESSERACT_CMD")
        if env_path and Path(env_path).exists():
            return env_path
        windows_default = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
        if windows_default.exists():
            return str(windows_default)
        return None

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        normalized = text.replace("€", "€")
        normalized = re.sub(r"e{2,}", "€", normalized)
        normalized = re.sub(r"(?<=\d)e(?=\d)", "€", normalized)
        normalized = re.sub(r"(?<=\d)e", "€", normalized)
        normalized = re.sub(r"e(?=\d)", "€", normalized)
        normalized = re.sub(r"(?<=\d)s(?=\d)", "5", normalized)
        normalized = re.sub(r"(?<=\d)o(?=\d)", "0", normalized)
        normalized = normalized.replace("€€", "€")
        return normalized
