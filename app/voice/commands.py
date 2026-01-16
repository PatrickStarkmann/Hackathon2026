"""Spoken command mappings."""

from __future__ import annotations

from typing import Dict


COMMANDS: Dict[str, str] = {
    "was ist das": "identify",
    "wie viele sind das": "count",
    "wieviele sind das": "count",
    "wieviel kostet das": "price",
    "wie viel kostet das": "price",
    "geldschein": "banknote",
}

KEY_COMMANDS: Dict[str, str] = {
    "i": "identify",
    "c": "count",
    "o": "obstacle",

    "b": "banknote",
    "p": "price",
}


def text_to_mode(text: str) -> str:
    if not text:
        return "idle"
    cleaned = " ".join(text.lower().strip().rstrip("?.!").split())
    return COMMANDS.get(cleaned, "idle")


def key_to_mode(key: int) -> str:
    if key == -1:
        return "idle"
    return KEY_COMMANDS.get(chr(key).lower(), "idle")
