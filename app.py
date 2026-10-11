from flask import Flask, render_template, request
from ultralytics import YOLO
import base64
import json
import os
from uuid import uuid4
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="")

load_dotenv(os.path.join(app.root_path, ".env"))

UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
app.config["GEMINI_MODEL"] = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# YOLO model
model = YOLO(os.path.join(app.root_path, "yolo11n.pt"))
animal_classifier = YOLO(os.path.join(app.root_path, "yolo11n-cls.pt"))

# Animal labels supported by the bundled YOLO COCO model.
ANIMAL_CLASSES = {
    "bird", "cat", "leopard", "jaguar", "dog", "lion", "deer", "horse", "sheep", "cow",
    "elephant", "bear", "tiger", "zebra", "giraffe"
}

RISK_LEVELS = {
    "leopard": "CRITICAL",
    "jaguar": "CRITICAL",
    "lion": "CRITICAL",
    "elephant": "CRITICAL",
    "tiger": "CRITICAL",
    "deer": "MEDIUM",
    "bear": "HIGH",
    "zebra": "HIGH",
    "giraffe": "HIGH",
    "cow": "MEDIUM",
    "horse": "MEDIUM",
    "dog": "MEDIUM",
    "bird": "LOW",
    "cat": "LOW",
    "sheep": "LOW"
}

RISK_PRIORITY = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return render_template("index.html", error="Choose an image or capture one with the camera."), 400

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="Choose an image or capture one with the camera."), 400

    safe_name = secure_filename(file.filename)
    extension = os.path.splitext(safe_name)[1].lower()
    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        return render_template("index.html", error="Please use a JPG, PNG, or WEBP image."), 400

    upload_dir = app.config["UPLOAD_FOLDER"]
    stored_filename = f"{uuid4().hex}{extension}"
    filepath = os.path.join(upload_dir, stored_filename)
    file.save(filepath)
    image_url = f"static/uploads/{stored_filename}"

    # Run YOLO detection, then refine visually similar animal classes.
    results = model(filepath)
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            animal = model.names[class_id]
            if animal in ANIMAL_CLASSES and confidence >= 0.40:
                if animal == "dog":
                    animal, confidence = classify_dog_detection(filepath, box)
                elif animal == "cat":
                    animal, confidence = classify_cat_detection(filepath, box)
                elif animal == "zebra":
                    animal, confidence = classify_zebra_detection(filepath, box)
                elif animal == "giraffe":
                    animal, confidence = classify_giraffe_detection(filepath, box)
                risk = RISK_LEVELS.get(animal, "MEDIUM")
                detections.append({
                    "animal": animal,
                    "confidence": round(confidence * 100, 2),
                    "risk": risk
                })

    if app.config["GEMINI_API_KEY"]:
        gemini_detections = call_gemini_for_animals(filepath)
        if gemini_detections is None:
            prediction_note = (
                "Gemini could not analyze this image. Showing local YOLO predictions instead; "
                "check the server log for details."
            )
        else:
            gemini_animals = {item["animal"] for item in gemini_detections}
            detections = [
                item for item in detections
                if item["animal"] in gemini_animals
            ]
            prediction_note = (
                "Only animal labels identified by both Gemini and the local YOLO model are shown. "
                "Disagreement can cause missed animals; verify high-risk detections visually."
            )
    else:
        prediction_note = (
            "Gemini is not configured; detections use the local YOLO models. "
            "AI confidence is not a validated probability."
        )

    alert_level = None
    alert_message = None
    crop_impact_message = None
    alert_animals = sorted({item["animal"].capitalize() for item in detections})
    highest_risk = max(
        (item["risk"] for item in detections),
        key=RISK_PRIORITY.__getitem__,
        default="LOW",
    )
    if highest_risk == "CRITICAL":
        alert_level = highest_risk
        alert_message = "Keep a safe distance and move people and livestock to shelter."
        crop_impact_message = (
            "Critical potential risk to crops and farmer yield: this wildlife may damage "
            "crops or threaten livestock, reducing harvest and farm income. Check nearby "
            "fields promptly; this image cannot confirm actual damage or predict yield loss."
        )
    elif highest_risk == "HIGH":
        alert_level = highest_risk
        alert_message = "Potentially dangerous wildlife detected. Keep your distance."
        crop_impact_message = (
            "High potential risk to crops and farm operations. Inspect nearby fields and "
            "protect crops and livestock; actual yield impact depends on the animal's "
            "behavior and proximity."
        )
    elif highest_risk == "MEDIUM":
        alert_level = highest_risk
        alert_message = "Use caution and keep a safe distance from the detected animal."
        crop_impact_message = (
            "Possible crop or property damage. Monitor the field and take sensible "
            "precautions; this scan does not establish that crops have been damaged."
        )
    elif highest_risk == "LOW":
        alert_level = highest_risk
        alert_message = "Low-risk animals were identified. Stay aware and avoid disturbing them."
        crop_impact_message = (
            "Low direct crop-risk rating for the detected animals. Continue routine "
            "monitoring; this image does not estimate crop yield."
        )

    return render_template(
        "result.html",
        image=image_url,
        detections=detections,
        alert_level=alert_level,
        alert_message=alert_message,
        crop_impact_message=crop_impact_message,
        alert_animals=alert_animals,
        prediction_note=prediction_note,
    )


def classify_zebra_detection(filepath, box):
    """Use the detected animal crop to distinguish a tiger from a zebra."""
    from PIL import Image

    with Image.open(filepath) as image:
        left, top, right, bottom = [int(value) for value in box.xyxy[0].tolist()]
        padding_x = int((right - left) * 0.1)
        padding_y = int((bottom - top) * 0.1)
        left = max(0, left - padding_x)
        top = max(0, top - padding_y)
        right = min(image.width, right + padding_x)
        bottom = min(image.height, bottom + padding_y)
        animal_crop = image.convert("RGB").crop((left, top, right, bottom))

    result = animal_classifier(animal_crop, verbose=False)[0]
    class_name = animal_classifier.names[int(result.probs.top1)]
    class_confidence = float(result.probs.top1conf)
    if class_name == "tiger" and class_confidence >= 0.5:
        return "tiger", class_confidence
    return "zebra", float(box.conf[0])


def classify_dog_detection(filepath, box):
    """Use the detected crop to distinguish lions from dogs."""
    from PIL import Image

    with Image.open(filepath) as image:
        left, top, right, bottom = [int(value) for value in box.xyxy[0].tolist()]
        padding_x = int((right - left) * 0.1)
        padding_y = int((bottom - top) * 0.1)
        left = max(0, left - padding_x)
        top = max(0, top - padding_y)
        right = min(image.width, right + padding_x)
        bottom = min(image.height, bottom + padding_y)
        animal_crop = image.convert("RGB").crop((left, top, right, bottom))

    result = animal_classifier(animal_crop, verbose=False)[0]
    class_name = animal_classifier.names[int(result.probs.top1)]
    class_confidence = float(result.probs.top1conf)
    if class_name == "lion" and class_confidence >= 0.5:
        return "lion", class_confidence
    return "dog", float(box.conf[0])


def classify_cat_detection(filepath, box):
    """Refine generic cat detections into leopard or jaguar when supported."""
    from PIL import Image

    with Image.open(filepath) as image:
        left, top, right, bottom = [int(value) for value in box.xyxy[0].tolist()]
        padding_x = int((right - left) * 0.25)
        padding_y = int((bottom - top) * 0.25)
        left = max(0, left - padding_x)
        top = max(0, top - padding_y)
        right = min(image.width, right + padding_x)
        bottom = min(image.height, bottom + padding_y)
        animal_crop = image.convert("RGB").crop((left, top, right, bottom))

    result = animal_classifier(animal_crop, verbose=False)[0]
    class_name = animal_classifier.names[int(result.probs.top1)]
    class_confidence = float(result.probs.top1conf)
    if class_name in {"leopard", "snow_leopard"} and class_confidence >= 0.2:
        return "leopard", class_confidence
    if class_name == "jaguar" and class_confidence >= 0.2:
        return "jaguar", class_confidence
    return "cat", float(box.conf[0])


def classify_giraffe_detection(filepath, box):
    """Check giraffe detections for deer-like ImageNet classifications."""
    from PIL import Image

    with Image.open(filepath) as image:
        left, top, right, bottom = [int(value) for value in box.xyxy[0].tolist()]
        padding_x = int((right - left) * 0.5)
        padding_y = int((bottom - top) * 0.5)
        left = max(0, left - padding_x)
        top = max(0, top - padding_y)
        right = min(image.width, right + padding_x)
        bottom = min(image.height, bottom + padding_y)
        animal_crop = image.convert("RGB").crop((left, top, right, bottom))

    result = animal_classifier(animal_crop, verbose=False)[0]
    class_name = animal_classifier.names[int(result.probs.top1)]
    class_confidence = float(result.probs.top1conf)
    deer_like_classes = {"gazelle", "hartebeest", "impala"}
    if class_name in deer_like_classes and class_confidence >= 0.2:
        return "deer", class_confidence
    return "giraffe", float(box.conf[0])


def call_gemini_for_animals(filepath):
    """Identify animals with Gemini; return None when the request or response fails."""
    api_key = app.config["GEMINI_API_KEY"]
    if not api_key:
        return None

    with open(filepath, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    mime_type = "image/jpeg"
    if filepath.lower().endswith(".png"):
        mime_type = "image/png"
    elif filepath.lower().endswith(".webp"):
        mime_type = "image/webp"

    payload = {
        "contents": [{
            "parts": [
                {
                    "text": "Identify every animal visible in this image. Return only valid JSON with this exact shape: {\"animals\":[{\"name\":\"animal_name\",\"confidence\":0.0}]}. Use only these names: bird, cat, leopard, jaguar, dog, lion, deer, horse, sheep, cow, elephant, bear, tiger, zebra, giraffe."
                },
                {"inline_data": {"mime_type": mime_type, "data": image_data}},
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0,
        },
    }

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{app.config['GEMINI_MODEL']}:generateContent?key={api_key}"
    )
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            raw_response = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        app.logger.warning("Gemini API request failed with HTTP status %s.", error.code)
        return None
    except (URLError, TimeoutError):
        app.logger.warning("Gemini API request could not reach the service.")
        return None
    except (ValueError, json.JSONDecodeError):
        app.logger.warning("Gemini API returned an invalid JSON response.")
        return None

    try:
        animals = raw_response["candidates"][0]["content"]["parts"][0]["text"]
        animals = json.loads(animals)
    except (KeyError, IndexError, TypeError, ValueError):
        app.logger.warning("Gemini API response did not contain the expected animal data.")
        return None

    if not isinstance(animals, dict) or not isinstance(animals.get("animals"), list):
        app.logger.warning("Gemini API response did not contain an animals list.")
        return None

    detections = []
    for animal in animals.get("animals", []):
        if not isinstance(animal, dict):
            continue
        name = str(animal.get("name", "")).strip().lower()
        try:
            confidence = float(animal.get("confidence", 0))
        except (TypeError, ValueError):
            continue
        if name in ANIMAL_CLASSES and 0 <= confidence <= 1:
            detections.append({
                "animal": name,
                "confidence": round(confidence * 100, 2),
                "risk": RISK_LEVELS.get(name, "MEDIUM"),
            })
    return detections


if __name__ == "__main__":
    app.run(debug=True)