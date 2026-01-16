"""Shared dataclasses for detections and decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Detection:
    label: str
    conf: float
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=True)
class Decision:
    text_to_say: str
    debug_text: str
    conf: float = 0.0
