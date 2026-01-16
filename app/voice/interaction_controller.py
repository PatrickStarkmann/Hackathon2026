# interaction_controller.py
from __future__ import annotations

from app.voice.speech_command_parser import SpeechCommandParser
from app.voice.speech_formatter import SpeechFormatter
from app.speech_module import SpeechEngine


class InteractionController:
    """
    Connects speech input -> intent -> formatted speech output.
    """

    def __init__(self, speech: SpeechEngine) -> None:
        self.speech = speech

    def handle_spoken_text(
        self,
        spoken_text: str,
        *,
        gegenstand: str,
        anzahl: int | None = None,
        preis_cent: int | None = None,
        position_text: str | None = None,
    ) -> None:
        command = SpeechCommandParser.parse(spoken_text)
        text = self.format_for_command(
            command,
            gegenstand=gegenstand,
            anzahl=anzahl,
            preis_cent=preis_cent,
            position_text=position_text,
        )
        self.speech.speak(text or "Das habe ich nicht verstanden.")

    def format_for_command(
        self,
        command: str,
        *,
        gegenstand: str,
        anzahl: int | None = None,
        preis_cent: int | None = None,
        position_text: str | None = None,
    ) -> str | None:
        if command == "banknote":
            try:
                return SpeechFormatter.banknote(int(gegenstand))
            except (TypeError, ValueError):
                return None
        if command == "identify":
            return SpeechFormatter.identify(gegenstand, position_text)
        if command == "count":
            if anzahl is None:
                return None
            return SpeechFormatter.count(gegenstand, anzahl)
        if command == "price":
            if preis_cent is None:
                return None
            return SpeechFormatter.price(gegenstand, preis_cent)
        if command == "full":
            return SpeechFormatter.full(gegenstand, anzahl, preis_cent)
        return None
