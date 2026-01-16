# speech_formatter.py
from __future__ import annotations


class SpeechFormatter:
    """
    Creates clear, blind-friendly speech output strings
    from structured vision results.
    """

    #--------------------------------------------------------------------------
    #Methode für die Identifikation eines Gegenstands
    @staticmethod
    def identify(gegenstand: str) -> str:
        return f"{gegenstand} erkannt."

    #--------------------------------------------------------------------------
    #Methode für die Mengenausgabe
    @staticmethod
    def count(gegenstand: str, anzahl: int) -> str:
        if anzahl == 1:
            return f"Ein {gegenstand}."
        return f"{anzahl} {gegenstand}."

    #--------------------------------------------------------------------------
    #Methode für die Preisansage
    @staticmethod
    def price(gegenstand: str, preis_cent: int) -> str:
        euro = preis_cent // 100
        cent = preis_cent % 100

        if cent == 0:
            return f"{gegenstand}. Preis {euro} Euro."
        return f"{gegenstand}. Preis {euro} Euro {cent} Cent."

    #--------------------------------------------------------------------------
    #Methode die alle Informationen kombiniert
    @staticmethod
    def full(gegenstand: str, anzahl: int | None = None, preis_cent: int | None = None) -> str:
        parts: list[str] = [gegenstand]

        if anzahl is not None:
            parts.append("ein Stück" if anzahl == 1 else f"{anzahl} Stück")

        if preis_cent is not None:
            euro = preis_cent // 100
            cent = preis_cent % 100
            if cent == 0:
                parts.append(f"Preis: {euro} Euro")
            else:
                parts.append(f"Preis: {euro} Euro {cent:02d}")

        return ", ".join(parts) + "."

