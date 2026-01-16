from speech_module import SpeechEngine, SpeechConfig
from speech_formatter import SpeechFormatter
import time

speech = SpeechEngine(SpeechConfig(cooldown_seconds=0.0))


speech.speak(SpeechFormatter.from_data("Banane"))
speech.speak(SpeechFormatter.from_data("Banane", preis_cent=89))
speech.speak(SpeechFormatter.from_data("Banane", anzahl=2, preis_cent=89))


time.sleep(10)   # wichtig!
speech.shutdown()
