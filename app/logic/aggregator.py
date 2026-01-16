"""Prediction stabilization helpers."""

from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Optional


class VoteAggregator:
    """Ring-buffer voting aggregator for stable labels."""

    def __init__(self, maxlen: int) -> None:
        self._buffer: Deque[str] = deque(maxlen=maxlen)

    def add(self, label: str) -> None:
        self._buffer.append(label)

    def majority(self, min_votes: int) -> Optional[str]:
        if len(self._buffer) < min_votes:
            return None
        counts = Counter(self._buffer)
        label, votes = counts.most_common(1)[0]
        if votes >= min_votes:
            return label
        return None

    def last(self) -> Optional[str]:
        if not self._buffer:
            return None
        return self._buffer[-1]

    def clear(self) -> None:
        self._buffer.clear()
