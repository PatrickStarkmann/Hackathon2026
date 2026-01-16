from speech_module import SpeechEngine, SpeechConfig
from speech_formatter import SpeechFormatter
import time

speech = SpeechEngine(SpeechConfig(cooldown_seconds=0.0))


print(SpeechFormatter.full("Banane", 2, 89))

speech.speak(SpeechFormatter.full("Banane", 2, 89))

time.sleep(10)   # wichtig!
speech.shutdown()
