# interaction_controller.py
from __future__ import annotations

from speech_command_parser import SpeechCommandParser
from speech_formatter import SpeechFormatter
from speech_module import SpeechEngine


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
    ) -> None:
        command = SpeechCommandParser.parse(spoken_text)

        if command == "identify":
            text = SpeechFormatter.from_data(
                gegenstand=gegenstand
            )

        elif command == "count":
            text = SpeechFormatter.from_data(
                gegenstand=gegenstand,
                anzahl=anzahl,
            )

        elif command == "price":
            text = SpeechFormatter.from_data(
                gegenstand=gegenstand,
                preis_cent=preis_cent,
            )

        elif command == "full":
            text = SpeechFormatter.from_data(
                gegenstand=gegenstand,
                anzahl=anzahl,
                preis_cent=preis_cent,
            )

        else:
            text = "Das habe ich nicht verstanden."

        self.speech.speak(text)
