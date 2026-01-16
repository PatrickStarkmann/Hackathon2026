"""Offline OCR module for extracting relevant label text."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2


@dataclass
class OCRResult:
    text_to_say: str
    raw_text: str
    beverage_type: str
    carbonation: str
    conf: float
    debug_text: str


class TextOCREngine:
    _MIN_WORD_CONF = 40
    _SPEAK_CONF_THRESHOLD = 0.45
    _MAX_LINES_SHORT = 2
    _MAX_LINES_FULL = 6
    _MAX_CHARS_SHORT = 120
    _MAX_CHARS_FULL = 300

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._tesseract_available = self._check_tesseract()

    def extract_text(self, frame: Any, mode: str = "short") -> OCRResult:
        if frame is None:
            return self._safe_result("Kein Text sicher erkennbar. Bitte näher ran und ruhig halten.")
        if not self._tesseract_available:
            return OCRResult(
                text_to_say="Texterkennung nicht verfügbar.",
                raw_text="",
                beverage_type="Unbekannt",
                carbonation="unbekannt",
                conf=0.0,
                debug_text="pytesseract/tesseract not available",
            )
        try:
            variants = self._preprocess_variants(frame)
            ocr_outputs = []
            for name, image in variants:
                data = self._run_ocr_data(image)
                words = self._filter_words(data)
                lines = self._group_lines(words)
                score = self._score_lines(lines)
                ocr_outputs.append((name, words, lines, score, data))
            best = max(ocr_outputs, key=lambda item: item[3], default=None)
            if best is None:
                return self._safe_result("Kein Text sicher erkennbar. Bitte näher ran und ruhig halten.")
            variant, words, lines, _score, data = best
            selected_lines, conf = self._select_lines(lines, mode)
            raw_text = "\n".join(line["text"] for line in lines)
            beverage_type, carbonation, bev_debug = self.classify_beverage(raw_text)
            text_to_say = self._compose_text(
                selected_lines, beverage_type, carbonation, mode
            )
            if conf < self._SPEAK_CONF_THRESHOLD:
                return OCRResult(
                    text_to_say="Kein Text sicher erkennbar. Bitte näher ran und ruhig halten.",
                    raw_text=raw_text,
                    beverage_type="Unbekannt",
                    carbonation="unbekannt",
                    conf=conf,
                    debug_text=f"low_conf={conf:.2f} variant={variant} words={len(words)}",
                )
            debug_text = (
                f"variant={variant} words={len(words)} lines={len(lines)} "
                f"sel_lines={len(selected_lines)} conf={conf:.2f} {bev_debug}"
            )
            return OCRResult(
                text_to_say=text_to_say,
                raw_text=raw_text,
                beverage_type=beverage_type,
                carbonation=carbonation,
                conf=conf,
                debug_text=debug_text,
            )
        except Exception as exc:
            self._logger.error("Text OCR failed: %s", exc)
            return OCRResult(
                text_to_say="Texterkennung nicht verfügbar.",
                raw_text="",
                beverage_type="Unbekannt",
                carbonation="unbekannt",
                conf=0.0,
                debug_text=f"OCR error: {exc}",
            )

    def annotate_frame_with_ocr(self, frame: Any, ocr_words: List[Dict[str, Any]]) -> Any:
        for word in ocr_words:
            x, y, w, h = word["bbox"]
            conf = word["conf"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 255), 1)
            cv2.putText(
                frame,
                f"{conf:.0f}",
                (x, max(15, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 180, 255),
                1,
            )
        return frame

    def classify_beverage(self, raw_text: str) -> Tuple[str, str, str]:
        normalized = self._normalize_text(raw_text)
        hits = []
        schorle = self._has_any(normalized, ["apfelschorle", "apfel schorle", "schorle"])
        apfel = self._has_any(normalized, ["apfel", "apfelsaft"])
        saft = self._has_any(normalized, ["saft", "fruchtsaft", "nektar"])
        wasser = self._has_any(normalized, ["mineralwasser", "wasser", "quellwasser", "tafelwasser"])
        sprudel = self._has_any(
            normalized,
            [
                "sprudel",
                "sprudelnd",
                "classic",
                "medium",
                "mit kohlensaure",
                "kohlensaure",
                "carbonated",
                "sparkling",
            ],
        )
        still = self._has_any(
            normalized,
            ["still", "ohne kohlensaure", "non-carbonated", "naturelle"],
        )
        carbonation = "unbekannt"
        if sprudel:
            carbonation = "sprudelnd"
        elif still:
            carbonation = "still"
        if schorle and apfel:
            beverage = "Apfelschorle"
        elif schorle:
            beverage = "Apfelschorle"
        elif wasser:
            if carbonation == "sprudelnd":
                beverage = "Sprudelwasser"
            elif carbonation == "still":
                beverage = "Stilles Wasser"
            else:
                beverage = "Mineralwasser"
        elif saft:
            beverage = "Saft"
        else:
            beverage = "Unbekannt"
        if schorle:
            hits.append("schorle")
        if saft:
            hits.append("saft")
        if wasser:
            hits.append("wasser")
        if sprudel:
            hits.append("sprudel")
        if still:
            hits.append("still")
        return beverage, carbonation, f"hits={','.join(hits)}"

    def _preprocess_variants(self, frame: Any) -> List[Tuple[str, Any]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned_otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
        cleaned_adaptive = cv2.morphologyEx(
            adaptive, cv2.MORPH_CLOSE, kernel, iterations=1
        )
        return [("otsu", cleaned_otsu), ("adaptive", cleaned_adaptive)]

    def _run_ocr_data(self, image: Any) -> Dict[str, List[Any]]:
        import pytesseract

        config = "--oem 3 --psm 6"
        return pytesseract.image_to_data(
            image, config=config, output_type=pytesseract.Output.DICT
        )

    def _filter_words(self, data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        words: List[Dict[str, Any]] = []
        for i, text in enumerate(data.get("text", [])):
            token = (text or "").strip()
            if not token:
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                continue
            if conf < self._MIN_WORD_CONF:
                continue
            if not self._is_meaningful_token(token):
                continue
            words.append(
                {
                    "text": token,
                    "conf": conf,
                    "block": data.get("block_num", [0])[i],
                    "par": data.get("par_num", [0])[i],
                    "line": data.get("line_num", [0])[i],
                    "bbox": (
                        int(data.get("left", [0])[i]),
                        int(data.get("top", [0])[i]),
                        int(data.get("width", [0])[i]),
                        int(data.get("height", [0])[i]),
                    ),
                }
            )
        return words

    def _group_lines(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lines: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
        for word in words:
            key = (word["block"], word["par"], word["line"])
            lines.setdefault(key, []).append(word)
        line_items: List[Dict[str, Any]] = []
        for key, line_words in lines.items():
            line_words.sort(key=lambda w: w["bbox"][0])
            text = " ".join(w["text"] for w in line_words)
            conf = sum(w["conf"] for w in line_words) / max(1, len(line_words))
            line_items.append(
                {
                    "key": key,
                    "text": text,
                    "conf": conf,
                    "word_count": len(line_words),
                }
            )
        return line_items

    def _score_lines(self, lines: List[Dict[str, Any]]) -> float:
        if not lines:
            return 0.0
        best = max(
            lines,
            key=lambda line: line["conf"] * max(1, len(line["text"].strip())),
        )
        return best["conf"] / 100.0

    def _select_lines(
        self, lines: List[Dict[str, Any]], mode: str
    ) -> Tuple[List[Dict[str, Any]], float]:
        if not lines:
            return [], 0.0
        sorted_lines = sorted(
            lines,
            key=lambda line: line["conf"] * max(1, len(line["text"].strip())),
            reverse=True,
        )
        max_lines = self._MAX_LINES_SHORT if mode == "short" else self._MAX_LINES_FULL
        max_chars = self._MAX_CHARS_SHORT if mode == "short" else self._MAX_CHARS_FULL
        selected: List[Dict[str, Any]] = []
        total_chars = 0
        for line in sorted_lines:
            if len(selected) >= max_lines:
                break
            line_text = line["text"].strip()
            if not line_text:
                continue
            if total_chars + len(line_text) > max_chars:
                continue
            selected.append(line)
            total_chars += len(line_text)
        conf = (
            sum(line["conf"] for line in selected) / max(1, len(selected))
        ) / 100.0
        if len(selected) < 1 or total_chars < 6:
            conf *= 0.6
        return selected, conf

    def _compose_text(
        self,
        selected_lines: List[Dict[str, Any]],
        beverage_type: str,
        carbonation: str,
        mode: str,
    ) -> str:
        lines_text = " ".join(line["text"] for line in selected_lines).strip()
        if beverage_type != "Unbekannt":
            if carbonation != "unbekannt":
                beverage_text = f"{beverage_type}. {carbonation.capitalize()}."
            else:
                beverage_text = f"{beverage_type}."
            if mode == "short" and lines_text:
                if beverage_text.lower() in lines_text.lower():
                    return beverage_text
                return f"{beverage_text} {lines_text}"
            if mode == "full" and lines_text:
                return f"{beverage_text} {lines_text}"
            return beverage_text
        return lines_text

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        normalized = text.lower()
        normalized = normalized.replace("ä", "ae")
        normalized = normalized.replace("ö", "oe")
        normalized = normalized.replace("ü", "ue")
        normalized = normalized.replace("ß", "ss")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _has_any(self, text: str, needles: List[str]) -> bool:
        for needle in needles:
            if needle in text:
                return True
        return False

    def _is_meaningful_token(self, token: str) -> bool:
        if len(token) < 2:
            return False
        alnum_count = sum(1 for ch in token if ch.isalnum())
        return alnum_count >= max(2, len(token) // 2)

    def _safe_result(self, text: str) -> OCRResult:
        return OCRResult(
            text_to_say=text,
            raw_text="",
            beverage_type="Unbekannt",
            carbonation="unbekannt",
            conf=0.0,
            debug_text="safe_return",
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
