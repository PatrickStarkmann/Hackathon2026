# speech_module_test.py
"""
Interactive keyboard test for SpeechEngine.

Keys:
  I -> Identify
  C -> Count
  O -> Obstacle
  B -> Banknote
  R -> Repeat last
  S -> Stop current speech
  Q -> Quit
"""

from __future__ import annotations

import sys
import time

from speech_module import SpeechEngine, SpeechConfig


def main() -> None:
    print("\n=== SpeechEngine Keyboard Test ===")
    print("I = Identify | C = Count | O = Obstacle | B = Banknote")
    print("R = Repeat last | S = Stop | Q = Quit\n")

    speech = SpeechEngine(
        SpeechConfig(
            cooldown_seconds=2.0,   # test cooldown clearly
            rate=170,               # adjust if you want
            volume=1.0,
        )
    )

    try:
        while True:
            key = input("> ").strip().lower()

            if key == "i":
                speech.speak("Apfel. Zweiundneunzig Prozent. Mitte.")

            elif key == "c":
                speech.speak("Ich sehe drei Artikel.")

            elif key == "o":
                speech.speak("Achtung. Hindernis links.")

            elif key == "b":
                speech.speak("Zehn Euro Schein erkannt.")

            elif key == "r":
                print("[repeat]")
                speech.repeat_last()

            elif key == "s":
                print("[stop]")
                speech.stop()

            elif key == "q":
                print("[quit]")
                break

            else:
                print("Unknown key")

            # Small sleep to simulate frame loop
            time.sleep(0.05)

    finally:
        speech.shutdown()
        print("SpeechEngine shut down cleanly.")


if __name__ == "__main__":
    main()
