# speech_formatter.py
from __future__ import annotations


class SpeechFormatter:
    """
    Creates clear, blind-friendly speech output strings
    from structured vision results.
    """

    # -------------------------------
    # Identify
    @staticmethod
    def identify(gegenstand: str) -> str:
        return f"{gegenstand} erkannt."

    # -------------------------------
    # Count
    @staticmethod
    def count(gegenstand: str, anzahl: int) -> str:
        if anzahl == 1:
            return f"Ein {gegenstand}."
        return f"{anzahl} {gegenstand}."

    # -------------------------------
    # Price
    @staticmethod
    def price(gegenstand: str, preis_cent: int) -> str:
        euro = preis_cent // 100
        cent = preis_cent % 100

        if cent == 0:
            return f"{gegenstand}. Preis {euro} Euro."
        return f"{gegenstand}. Preis {euro} Euro {cent} Cent."

    # -------------------------------
    # Full
    @staticmethod
    def full(
        gegenstand: str,
        anzahl: int | None = None,
        preis_cent: int | None = None,
    ) -> str:
        parts: list[str] = [gegenstand]

        if anzahl is not None:
            parts.append("ein Stück" if anzahl == 1 else f"{anzahl} Stück")

        if preis_cent is not None:
            euro = preis_cent // 100
            cent = preis_cent % 100
            if cent == 0:
                parts.append(f"Preis {euro} Euro")
            else:
                parts.append(f"Preis {euro} Euro {cent:02d}")

        return ", ".join(parts) + "."

    # -------------------------------
    # AUTO DISPATCH 
    @staticmethod
    def from_data(
        gegenstand: str,
        anzahl: int | None = None,
        preis_cent: int | None = None,
    ) -> str:
        """
        Automatically decide what to speak based on available data.
        """
        if preis_cent is not None and anzahl is not None:
            return SpeechFormatter.full(gegenstand, anzahl, preis_cent)

        if preis_cent is not None:
            return SpeechFormatter.price(gegenstand, preis_cent)

        if anzahl is not None:
            return SpeechFormatter.count(gegenstand, anzahl)

        return SpeechFormatter.identify(gegenstand)
