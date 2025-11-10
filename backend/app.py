import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import json

# Optional dependencies; handle absence gracefully at runtime
try:
    import joblib  # for scikit-learn model
except Exception:  # pragma: no cover
    joblib = None

try:
    from tensorflow.keras.models import load_model
except Exception:  # pragma: no cover
    load_model = None

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

YIELD_MODEL_PATH = os.path.join(MODELS_DIR, "yield_model.pkl")
HEALTH_MODEL_PATH = os.path.join(MODELS_DIR, "crop_health_model.h5")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")

yield_model = None
health_model = None
health_labels = None


def _load_yield_model():
    global yield_model
    if os.path.exists(YIELD_MODEL_PATH) and joblib is not None:
        try:
            yield_model = joblib.load(YIELD_MODEL_PATH)
        except Exception:
            yield_model = None


def _load_health_model():
    global health_model
    if os.path.exists(HEALTH_MODEL_PATH) and load_model is not None:
        try:
            health_model = load_model(HEALTH_MODEL_PATH)
        except Exception:
            health_model = None


def _load_health_labels():
    global health_labels
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                labels = json.load(f)
                # Expecting list or dict mapping index->name
                if isinstance(labels, list):
                    health_labels = labels
                elif isinstance(labels, dict):
                    # Normalize keys to int order if possible
                    try:
                        max_idx = max(int(k) for k in labels.keys())
                        health_labels = [labels.get(str(i), str(i)) for i in range(max_idx + 1)]
                    except Exception:
                        health_labels = None
                else:
                    health_labels = None
        except Exception:
            health_labels = None


_load_yield_model()
_load_health_model()
_load_health_labels()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "yield_model_loaded": bool(yield_model),
        "health_model_loaded": bool(health_model),
        "health_labels_loaded": bool(health_labels),
    })


@app.route("/api/predict-yield", methods=["POST"])
def predict_yield():
    if yield_model is None:
        return jsonify({"error": "Yield model not available. Place model at 'backend/models/yield_model.pkl'."}), 503

    data = request.get_json(silent=True) or {}
    required = ["rainfall", "temperature", "soil_ph", "fertilizer", "area"]

    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        vals = [
            float(data["rainfall"]),
            float(data["temperature"]),
            float(data["soil_ph"]),
            float(data["fertilizer"]),
            float(data["area"]),
        ]
    except (TypeError, ValueError):
        return jsonify({"error": "All input features must be numeric."}), 400

    try:
        X = np.array([vals])
        pred = float(yield_model.predict(X)[0])
        return jsonify({"predicted_yield": round(pred, 2)})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/predict-health", methods=["POST"])
def predict_health():
    if health_model is None:
        return jsonify({"error": "Health model not available. Place model at 'backend/models/crop_health_model.h5'."}), 503

    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No file uploaded. Use form-data with key 'image'."}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((128, 128))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = health_model.predict(arr)
        preds = np.array(preds)
        if preds.ndim == 2:
            probs = preds[0]
        else:
            probs = preds.squeeze()
        label = int(np.argmax(probs))
        confidence = float(np.max(probs))

        resp = {
            "label": label,
            "confidence": round(confidence, 4)
        }
        if health_labels and 0 <= label < len(health_labels):
            resp["label_name"] = health_labels[label]
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": f"Health prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
