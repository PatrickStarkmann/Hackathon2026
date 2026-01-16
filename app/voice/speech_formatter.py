# speech_formatter.py
from __future__ import annotations


class SpeechFormatter:
    """
    Creates clear, blind-friendly speech output strings
    from structured vision results.
    """

    @staticmethod
    def identify(gegenstand: str) -> str:
        return f"{gegenstand} erkannt."

    @staticmethod
    def count(gegenstand: str, anzahl: int) -> str:
        if anzahl == 1:
            return f"Ein {gegenstand}."
        return f"{anzahl} {gegenstand}."

    @staticmethod
    def price(gegenstand: str, preis_cent: int) -> str:
        euro = preis_cent // 100
        cent = preis_cent % 100

        if cent == 0:
            return f"{gegenstand}. Preis {euro} Euro."
        return f"{gegenstand}. Preis {euro} Euro {cent} Cent."

    @staticmethod
    def full(
        gegenstand: str,
        anzahl: int | None = None,
        preis_cent: int | None = None,
    ) -> str:
        """
        Combined output, used for demos or banknote-like info.
        """
        parts: list[str] = [f"{gegenstand}."]

        if anzahl is not None:
            if anzahl == 1:
                parts.append("Ein Stück.")
            else:
                parts.append(f"{anzahl} Stück.")

        if preis_cent is not None:
            euro = preis_cent // 100
            cent = preis_cent % 100
            if cent == 0:
                parts.append(f"{euro} Euro.")
            else:
                parts.append(f"{euro} Euro {cent} Cent.")

        return " ".join(parts)
