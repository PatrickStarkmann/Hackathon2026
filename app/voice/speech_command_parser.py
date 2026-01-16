# speech_command_parser.py
from __future__ import annotations


class SpeechCommandParser:
    """
    Maps spoken user commands to system actions.
    Offline, rule-based, demo-stable.
    """

    IDENTIFY_KEYWORDS = [
        "was ist das",
        "was ist das für ein",
        "was ist das fuer ein",
        "was sehe ich",
        "erkennen",
    ]

    COUNT_KEYWORDS = [
        "wie viele",
        "wieviele",
        "anzahl",
        "wie oft",
    ]

    PRICE_KEYWORDS = [
        "was kostet",
        "wie viel kostet",
        "preis",
        "kosten",
    ]

    FULL_KEYWORDS = [
        "alles",
        "alle informationen",
        "voll",
    ]

    @staticmethod
    def parse(text: str) -> str:
        """
        Returns one of:
        - identify
        - count
        - price
        - full
        - unknown
        """
        if not text:
            return "unknown"

        t = text.lower().strip()

        for kw in SpeechCommandParser.IDENTIFY_KEYWORDS:
            if kw in t:
                return "identify"

        for kw in SpeechCommandParser.COUNT_KEYWORDS:
            if kw in t:
                return "count"

        for kw in SpeechCommandParser.PRICE_KEYWORDS:
            if kw in t:
                return "price"

        for kw in SpeechCommandParser.FULL_KEYWORDS:
            if kw in t:
                return "full"

        return "unknown"
