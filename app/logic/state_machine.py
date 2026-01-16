"""Command state handling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CommandState:
    mode: str = "idle"

    def set_mode(self, mode: str) -> None:
        self.mode = mode
