# Blind Shopping Assistant (Offline)

Offline, laptop-friendly assistant for hackathon demos. Built with OpenCV, YOLOv8, and pyttsx3.

## Features (MVP)
- Keyboard-triggered commands: Identify, Count, Obstacle, Banknote, Price.
- Stabilized predictions via voting to reduce flicker.
- Cooldown to prevent audio spam.
- Debug window with bounding boxes and FPS (toggle via config).
- Fully offline runtime (no API calls).

## Project Structure
```
app/
  main.py
  camera_module.py
  vision_module.py
  logic_module.py
  speech_module.py
  config.py
  common.py
  banknote/
    banknote_module.py
    tflite_stub.py
    ocr_stub.py
  voice/
    commands.py
    stt_vosk_stub.py
  logic/
    aggregator.py
    state_machine.py
assets/
  (place model weights here)
requirements.txt
smoke_test.py
```

## Setup (macOS/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m app.main
```

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m app.main
```

## Model Weights
- Place YOLOv8 weights at `assets/yolo_weights.pt`.
- If the file is missing, the app runs in dummy mode (fake detections for integration).
- Optional banknote model: `assets/banknote.tflite` and labels at `assets/banknote_labels.txt` (kein geld, 5, 10, 20, 50, 100).
- Best practice: export a Teachable Machine image classifier (TFLite) with exactly those 6 classes in that order.

## Banknote Setup (Real Recognition)
1) Export a TFLite image classifier (Teachable Machine recommended).
2) Copy the model to `assets/banknote.tflite`.
3) Ensure `assets/banknote_labels.txt` matches the model class order:
   `kein geld, 5, 10, 20, 50, 100`
4) Install a TFLite runtime:
   - Recommended: `pip install -r requirements-banknote.txt`

## Optional Offline STT (Vosk)
- `vosk` is included in `requirements.txt` to simplify future offline speech-to-text integration.
- Current default input is keyboard-only; see `app/voice/stt_vosk_stub.py` for the stub interface.

## Key Bindings (Demo)
- **I**: Identify (“Was ist das?”)
- **C**: Count (“Wie viele siehst du?”)
- **O**: Obstacle (“Ist etwas vor mir?”)
- **B**: Banknote (“Welche Banknote ist das?”) — TFLite optional, OCR fallback
- **P**: Price (“Wie teuer ist das?”) — OCR optional
- **Q / ESC**: Quit

## Troubleshooting
- **Camera permissions**: Ensure the OS has granted camera access to Python.
- **Missing weights**: Place `yolo_weights.pt` in `assets/` or use dummy mode.
- **No audio**: Verify speakers, volume, and pyttsx3 backend support.
- **Price/Banknote OCR**: Install `pytesseract` + system Tesseract if you want OCR-based fallback.
- **Low confidence**: Improve lighting or move closer to the object.

## Smoke Test
Run lightweight checks to ensure camera, model loading, and TTS are working:
```bash
python smoke_test.py
```
