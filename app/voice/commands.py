"""Keyboard command mappings."""

from __future__ import annotations

from typing import Dict


COMMANDS: Dict[str, str] = {
    "i": "identify",
    "c": "count",
    "o": "obstacle",
    "b": "banknote",
    "p": "price",
}


def key_to_mode(key: int) -> str:
    if key == -1:
        return "idle"
    char = chr(key).lower()
    return COMMANDS.get(char, "idle")
