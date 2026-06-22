"""
Skin Cancer Detection - Flask app (pure HTML/CSS, no JS framework).

A single Flask server that:
  * serves an HTML upload form at  GET  /
  * runs the model and renders results server-side at  POST /
  * exposes a JSON API at  POST /api/predict  and a  GET /health  check.

NOTE: This is an educational/screening-aid demo. It is NOT a medical device
and must not be used for actual diagnosis. Always consult a dermatologist.
"""

import base64
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "model"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
MAX_FILE_MB = 10

# HAM10000 class labels (7 classes). Order MUST match the training generator's
# class_indices (alphabetical for the 224px demo models).
CLASS_LABELS = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MALIGNANT = {"mel", "bcc", "akiec"}

# ---------------------------------------------------------------------------
# Metadata encoding (multimodal model) - must match the training notebook.
# Vector = [age/100] + one-hot(sex) + one-hot(localization) = 19 dims.
# ---------------------------------------------------------------------------
SEX_CATEGORIES = ["female", "male", "unknown"]
LOC_CATEGORIES = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
    "hand", "lower extremity", "neck", "scalp", "trunk", "unknown",
    "upper extremity",
]
AGE_FILL = 50.0
META_DIM = 1 + len(SEX_CATEGORIES) + len(LOC_CATEGORIES)  # 19


def encode_meta(age: Optional[float], sex: Optional[str], loc: Optional[str]):
    """Build the 19-d metadata vector exactly as the notebook does."""
    out = []
    try:
        a = float(age) if age not in (None, "") else AGE_FILL
        if np.isnan(a):
            a = AGE_FILL
    except (TypeError, ValueError):
        a = AGE_FILL
    out.append(min(max(a, 0.0), 100.0) / 100.0)

    s = str(sex).lower().strip() if sex else "unknown"
    s = s if s in SEX_CATEGORIES else "unknown"
    out += [1.0 if s == c else 0.0 for c in SEX_CATEGORIES]

    l = str(loc).lower().strip() if loc else "unknown"
    l = l if l in LOC_CATEGORIES else "unknown"
    out += [1.0 if l == c else 0.0 for c in LOC_CATEGORIES]

    return np.array([out], dtype=np.float32)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

_model = None  # lazily loaded singleton


def load_model():
    """Load the TF model once. Returns None if no model is present yet."""
    global _model
    if _model is not None:
        return _model

    import tensorflow as tf  # imported lazily so the app can boot without TF

    keras_path = MODEL_DIR / "model.keras"
    h5_path = MODEL_DIR / "model.h5"
    saved_model_path = MODEL_DIR / "saved_model"

    try:
        if keras_path.exists():
            _model = tf.keras.models.load_model(keras_path)
        elif h5_path.exists():
            _model = tf.keras.models.load_model(h5_path)
        elif saved_model_path.exists():
            _model = tf.keras.layers.TFSMLayer(
                str(saved_model_path), call_endpoint="serving_default"
            )
        else:
            return None
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load model: {exc}") from exc

    return _model


def model_expects_metadata(model) -> bool:
    try:
        return len(model.inputs) >= 2
    except Exception:
        return False


def model_image_size(model) -> int:
    try:
        for inp in model.inputs:
            shape = inp.shape
            if len(shape) == 4 and shape[1] is not None:
                return int(shape[1])
    except Exception:
        pass
    return IMG_SIZE


def preprocess(image_bytes: bytes, size: int) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def run_prediction(image_bytes: bytes, age, sex, localization) -> dict:
    """Core inference shared by the HTML form and the JSON API."""
    model = load_model()
    if model is None:
        raise RuntimeError(
            "No model found. Place a trained model in backend/model/ "
            "(model.keras, model.h5, or saved_model/)."
        )

    batch = preprocess(image_bytes, model_image_size(model))
    if model_expects_metadata(model):
        inputs = [batch, encode_meta(age, sex, localization)]
    else:
        inputs = batch

    preds = model(inputs) if callable(model) else model.predict(inputs)
    if isinstance(preds, dict):
        preds = list(preds.values())[0]
    preds = np.asarray(preds).reshape(-1)

    top_idx = int(np.argmax(preds))
    top_key = CLASS_ORDER[top_idx]

    probabilities = [
        {
            "key": CLASS_ORDER[i],
            "label": CLASS_LABELS[CLASS_ORDER[i]],
            "probability": float(preds[i]),
            "percent": round(float(preds[i]) * 100, 1),
            "malignant": CLASS_ORDER[i] in MALIGNANT,
        }
        for i in range(len(CLASS_ORDER))
    ]
    probabilities.sort(key=lambda p: p["probability"], reverse=True)

    malignant_probability = float(sum(preds[CLASS_ORDER.index(k)] for k in MALIGNANT))
    if malignant_probability >= 0.5:
        message = "Possibly cancerous - please see a dermatologist."
        level = "high"
    elif malignant_probability >= 0.2:
        message = "Uncertain - some signs of a cancerous lesion. Get it checked."
        level = "medium"
    else:
        message = "Likely benign (not cancer) - but this is not a diagnosis."
        level = "low"

    return {
        "prediction": {
            "key": top_key,
            "label": CLASS_LABELS[top_key],
            "confidence": float(preds[top_idx]),
            "confidence_percent": round(float(preds[top_idx]) * 100, 1),
            "malignant": top_key in MALIGNANT,
        },
        "cancer_assessment": {
            "is_cancer_likely": malignant_probability >= 0.5,
            "malignant_probability": malignant_probability,
            "malignant_percent": round(malignant_probability * 100, 1),
            "benign_percent": round((1.0 - malignant_probability) * 100, 1),
            "level": level,
            "message": message,
        },
        "probabilities": probabilities,
        "disclaimer": (
            "This result is from an educational model and may be wrong. "
            "It is not a diagnosis. Please consult a qualified dermatologist."
        ),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def model_present() -> bool:
    """True if a model file exists on disk (regardless of whether it's loaded)."""
    return (
        (MODEL_DIR / "model.keras").exists()
        or (MODEL_DIR / "model.h5").exists()
        or (MODEL_DIR / "saved_model").exists()
    )


def _form_context(**overrides):
    ctx = {
        "loc_categories": LOC_CATEGORIES,
        "max_file_mb": MAX_FILE_MB,
        "model_loaded": _model is not None or model_present(),
        "age": "",
        "sex": "unknown",
        "localization": "unknown",
        "result": None,
        "error": None,
        "image_data_uri": None,
    }
    ctx.update(overrides)
    return ctx


@app.get("/")
def index():
    return render_template("index.html", **_form_context())


@app.post("/")
def predict_form():
    age = request.form.get("age", "")
    sex = request.form.get("sex", "unknown")
    localization = request.form.get("localization", "unknown")

    file = request.files.get("file")
    if file is None or file.filename == "":
        return render_template(
            "index.html",
            **_form_context(age=age, sex=sex, localization=localization,
                            error="Please choose an image file (JPG or PNG)."),
        )

    if not (file.mimetype or "").startswith("image/"):
        return render_template(
            "index.html",
            **_form_context(age=age, sex=sex, localization=localization,
                            error="The uploaded file must be an image (JPG or PNG)."),
        )

    image_bytes = file.read()
    try:
        result = run_prediction(image_bytes, age, sex, localization)
    except RuntimeError as exc:
        return render_template(
            "index.html",
            **_form_context(age=age, sex=sex, localization=localization, error=str(exc)),
        )
    except Exception:
        return render_template(
            "index.html",
            **_form_context(age=age, sex=sex, localization=localization,
                            error="Could not read that image. Try a different JPG or PNG."),
        )

    data_uri = "data:{};base64,{}".format(
        file.mimetype or "image/jpeg",
        base64.b64encode(image_bytes).decode("ascii"),
    )
    return render_template(
        "index.html",
        **_form_context(age=age, sex=sex, localization=localization,
                        result=result, image_data_uri=data_uri),
    )


@app.post("/api/predict")
def predict_api():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"detail": "No image file provided."}), 400
    if not (file.mimetype or "").startswith("image/"):
        return jsonify({"detail": "Uploaded file must be an image."}), 400
    try:
        result = run_prediction(
            file.read(),
            request.form.get("age"),
            request.form.get("sex"),
            request.form.get("localization"),
        )
    except RuntimeError as exc:
        return jsonify({"detail": str(exc)}), 503
    except Exception:
        return jsonify({"detail": "Invalid or unsupported image file."}), 400
    return jsonify(result)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": load_model() is not None})


@app.errorhandler(413)
def too_large(_):
    return render_template(
        "index.html",
        **_form_context(error=f"Image is too large. Max {MAX_FILE_MB} MB."),
    ), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "1") != "0"
    app.run(host="0.0.0.0", port=port, debug=debug)
