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
    def identify(gegenstand: str, position: str | None = None) -> str:
        if position:
            return f"{gegenstand} erkannt. {position}."
        return f"{gegenstand} erkannt."

    # -------------------------------
    # Count
    @staticmethod
    def count(gegenstand: str, anzahl: int) -> str:
        number_words = {
            1: "Ein",
            2: "Zwei",
            3: "Drei",
            4: "Vier",
            5: "Fuenf",
            6: "Sechs",
            7: "Sieben",
            8: "Acht",
            9: "Neun",
            10: "Zehn",
        }
        number_text = number_words.get(anzahl, str(anzahl))
        return f"{gegenstand}. {number_text} Stück."

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
    # Banknote
    @staticmethod
    def banknote(wert_euro: int) -> str:
        number_words = {
            1: "Ein",
            2: "Zwei",
            3: "Drei",
            4: "Vier",
            5: "Fuenf",
            6: "Sechs",
            7: "Sieben",
            8: "Acht",
            9: "Neun",
            10: "Zehn",
            20: "Zwanzig",
            50: "Fuenfzig",
            100: "Hundert",
        }
        number_text = number_words.get(wert_euro, str(wert_euro))
        return f"{number_text} Euro."

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
