# interaction_test.py
from __future__ import annotations

import time

from speech_module import SpeechEngine, SpeechConfig
from interaction_controller import InteractionController


def main() -> None:
    speech = SpeechEngine(SpeechConfig(cooldown_seconds=0.0, rate=170))
    controller = InteractionController(speech)

    # Dummy "Vision"-Daten (später ersetzt ihr das durch echte Inference)
    gegenstand = "Banane"
    anzahl = 2
    preis_cent = 89

    print("\n=== Interaction Test (Speech -> Intent -> TTS) ===")
    print("Sag (tippe) z.B.:")
    print('- "Was ist das?"')
    print('- "Wie viele sind das?"')
    print('- "Was kostet das?"')
    print('- "Zeig mir alles"')
    print('q zum Beenden\n')

    try:
        while True:
            spoken_text = input("> ").strip()
            if spoken_text.lower() in ("q", "quit", "exit"):
                break

            controller.handle_spoken_text(
                spoken_text,
                gegenstand=gegenstand,
                anzahl=anzahl,
                preis_cent=preis_cent,
            )

            # kurz Luft lassen, damit du nicht aus Versehen spamst
            time.sleep(0.05)

    finally:
        speech.shutdown()
        print("Bye!")


if __name__ == "__main__":
    main()
