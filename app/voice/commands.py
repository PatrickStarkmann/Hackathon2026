"""Spoken command mappings."""

from __future__ import annotations

from typing import Dict


COMMANDS: Dict[str, str] = {
    "was ist das": "identify",
    "wieviele sind das": "count",
    "o": "obstacle",
    "wieviel kostest das": "price",
}


def text_to_mode(text: str) -> str:
    if not text:
        return "idle"
    cleaned = " ".join(text.lower().strip().rstrip("?.!").split())
    return COMMANDS.get(cleaned, "idle")


def key_to_mode(key: int) -> str:
    if key == -1:
        return "idle"
    return text_to_mode(chr(key))
