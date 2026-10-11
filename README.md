# AgriGuard AI

AgriGuard AI is a Flask web application that analyzes farm images for wildlife and displays animal detections with potential risk levels. You can upload a photo or capture one with your device's camera.

> **Important:** This application is a prototype decision-support tool. AI predictions can be wrong and are not validated probabilities. Risk labels do not confirm crop damage or predict yield loss. Do not approach wildlife based only on an app result; follow local safety guidance.

## Features

- Detects supported animals in JPG, JPEG, PNG, and WEBP images.
- Refines some similar-looking detections using the bundled YOLO classification model.
- Reports detection confidence and a potential risk rating.
- Offers optional animal-label cross-checking with the Gemini API.
- Displays high-risk alerts and browser notification and sound controls.
- Accepts uploaded images up to 10 MB.

Supported labels: bird, cat, leopard, jaguar, dog, lion, deer, horse, sheep, cow, elephant, bear, tiger, zebra, and giraffe.

## Requirements

- Python installed on your computer.
- A modern browser. Camera access generally requires `localhost` or a secure HTTPS connection.
- A Gemini API key is optional. Without one, the app uses its local YOLO models.

## Setup

Run these commands from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

The YOLO model weights (`yolo11n.pt` and `yolo11n-cls.pt`) are included in the repository.

## Optional Gemini configuration

To enable Gemini cross-checking, copy `.env.example` to `.env` and set your own API key:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-latest
```

Keep `.env` private; it is excluded from Git. When Gemini responds successfully, the app shows only animal labels also detected by the local YOLO model. If Gemini is not configured or its request fails, the app displays local YOLO predictions.

## Run

With the virtual environment activated, start the Flask app:

```powershell
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. The app runs with Flask debug mode enabled for local development; do not expose this development server to the public internet.

## How it works

1. The Flask app accepts and saves an image in `static/uploads/`.
2. YOLO detects supported animal classes; a second bundled model refines selected visually similar classes.
3. If configured, Gemini can cross-check the animal labels.
4. The results page shows the animals, confidence scores, potential risk levels, and safety guidance.

Uploaded images are stored locally in `static/uploads/`, which is excluded from Git. Risk levels are application-defined estimates, not an assessment of actual animal behavior or farm conditions.

## Project files

- `app.py` — Flask routes, image processing, and AI integration.
- `index.html`, `result.html` — upload and results pages.
- `style.css`, `static/style.css`, `alert.js`, `static/alert.wav` — interface styling and alert behavior.
- `yolo11n.pt`, `yolo11n-cls.pt` — bundled YOLO model weights.
- `requirements.txt` — Python dependencies.
