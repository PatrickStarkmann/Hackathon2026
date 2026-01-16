"""TFLite banknote classifier (optional)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.common import Decision
from app.config import BANKNOTE_CONF_THRESHOLD, BANKNOTE_MARGIN, BANKNOTE_VOTE_MIN, BANKNOTE_VOTE_N
from app.logic.aggregator import VoteAggregator


class TfliteBanknoteStub:
    """TFLite classifier with graceful fallback when unavailable."""

    def __init__(
        self,
        model_path: str = "assets/banknote.tflite",
        labels_path: str = "assets/banknote_labels.txt",
        conf_threshold: float = BANKNOTE_CONF_THRESHOLD,
        vote_n: int = BANKNOTE_VOTE_N,
        vote_min: int = BANKNOTE_VOTE_MIN,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._model_path = Path(model_path)
        self._labels_path = Path(labels_path)
        self._conf_threshold = conf_threshold
        self._votes = VoteAggregator(maxlen=vote_n)
        self._vote_min = vote_min
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._labels = ["kein geld", "5", "10", "20", "50", "100"]
        self._expected_labels = {"kein geld", "5", "10", "20", "50", "100"}
        self._reason: Optional[str] = None
        self._last_label: Optional[str] = None

        if self._labels_path.exists():
            raw_labels = [line.strip() for line in self._labels_path.read_text().splitlines() if line.strip()]
            self._labels = [self._normalize_label(line) for line in raw_labels]
            missing = self._expected_labels.difference({label.lower() for label in self._labels})
            if missing:
                self._logger.warning("Banknote labels missing expected classes: %s", ", ".join(sorted(missing)))

        tflite = self._load_tflite_runtime()
        if tflite is None:
            self._reason = "tflite runtime missing"
            return
        if not self._model_path.exists():
            self._reason = f"model missing: {self._model_path}"
            return
        try:
            self._interpreter = tflite.Interpreter(model_path=str(self._model_path))
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            output_shape = self._output_details[0].get("shape")
            if output_shape is not None:
                output_shape = list(output_shape)
            if output_shape is not None and len(output_shape) > 0 and int(output_shape[-1]) != len(self._labels):
                self._logger.warning(
                    "Banknote label count (%s) does not match model output (%s).",
                    len(self._labels),
                    output_shape[-1],
                )
        except Exception as exc:
            self._reason = f"tflite init failed: {exc}"
            self._interpreter = None

    def predict(self, frame) -> Decision:
        if self._interpreter is None or self._input_details is None or self._output_details is None:
            reason = self._reason or "tflite unavailable"
            return Decision(text_to_say="", debug_text=f"tflite_unavailable: {reason}", conf=0.0)

        try:
            input_data = self._preprocess(frame)
            self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_details[0]["index"])
        except Exception as exc:
            self._logger.error("TFLite inference failed: %s", exc)
            return Decision(text_to_say="", debug_text="tflite_unavailable: tflite_error", conf=0.0)

        scores = output[0]
        out_info = self._output_details[0]
        out_scale, out_zero = out_info.get("quantization", (0.0, 0))
        if out_scale > 0:
            scores = (scores.astype(np.float32) - out_zero) * out_scale
        conf = float(np.max(scores))
        label_idx = int(np.argmax(scores))
        label = self._labels[label_idx] if label_idx < len(self._labels) else str(label_idx)
        top2 = np.argsort(scores)[-2:][::-1]
        top2_labels = []
        for idx in top2:
            name = self._labels[int(idx)] if int(idx) < len(self._labels) else str(int(idx))
            top2_labels.append(f"{name}:{float(scores[int(idx)]):.2f}")
        margin = float(scores[int(top2[0])] - scores[int(top2[1])]) if len(top2) > 1 else conf
        if self._last_label is not None and label != self._last_label and conf >= self._conf_threshold:
            self._votes.clear()
        self._last_label = label
        self._votes.add(label)
        stable = self._votes.majority(self._vote_min)
        if stable is None and conf >= self._conf_threshold and margin >= BANKNOTE_MARGIN:
            stable = label

        if conf < self._conf_threshold or stable is None or margin < BANKNOTE_MARGIN:
            text = "Unsicher. Bitte Schein flach halten und nah an die Kamera."
            debug = f"tflite_uncertain {label} {conf:.2f} m={margin:.2f} top={','.join(top2_labels)}"
            return Decision(text_to_say=text, debug_text=debug, conf=conf)

        stable_norm = stable.lower().replace(" ", "") if stable else ""
        if stable_norm in {"keingeld", "nogeld", "none", "nomoney", "0"}:
            return Decision(text_to_say="Kein Geld erkannt.", debug_text="tflite_none", conf=conf)

        text = f"{stable} Euro."
        debug = f"tflite {stable} {conf:.2f}"
        return Decision(text_to_say=text, debug_text=debug, conf=conf)

    def _load_tflite_runtime(self):
        try:
            import tflite_runtime.interpreter as tflite

            return tflite
        except Exception:
            try:
                import tensorflow.lite as tflite

                return tflite
            except Exception:
                return None

    def _preprocess(self, frame) -> np.ndarray:
        input_info = self._input_details[0]
        shape = input_info["shape"]
        dtype = input_info["dtype"]
        height, width = int(shape[1]), int(shape[2])

        import cv2

        resized = cv2.resize(frame, (width, height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        data = np.expand_dims(rgb, axis=0).astype(dtype)

        if dtype in (np.float32, np.float16):
            data = data / 255.0
        else:
            scale, zero = input_info.get("quantization", (0.0, 0))
            if scale > 0:
                data = data / scale + zero

        return data

    @staticmethod
    def _normalize_label(line: str) -> str:
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            return parts[1].strip()
        return line.strip()
