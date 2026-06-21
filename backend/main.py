"""
Skin Cancer Detection - FastAPI backend.

Exposes a /predict endpoint that accepts an uploaded image and returns the
predicted skin-lesion class with a confidence score.

NOTE: This is an educational/screening-aid demo. It is NOT a medical device
and must not be used for actual diagnosis. Always consult a dermatologist.
"""

import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "model"))
# Input size must match what the model was trained on (SqueezeNet -> 224).
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))

# HAM10000 class labels (7 classes). Order MUST match the training generator's
# class_indices. The training notebook prints this mapping; keep it in sync.
CLASS_LABELS = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
# Index order used during training (alphabetical, as Keras does by default).
CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
# Classes considered malignant / requiring urgent attention.
MALIGNANT = {"mel", "bcc", "akiec"}

# ---------------------------------------------------------------------------
# Metadata encoding (multimodal model)
# ---------------------------------------------------------------------------
# These MUST match the constants in notebook/train_skin_cancer.ipynb. The
# metadata vector is [age/100] + one-hot(sex) + one-hot(localization) = 19.
SEX_CATEGORIES = ["female", "male", "unknown"]
LOC_CATEGORIES = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
    "hand", "lower extremity", "neck", "scalp", "trunk", "unknown",
    "upper extremity",
]
AGE_FILL = 50.0
META_DIM = 1 + len(SEX_CATEGORIES) + len(LOC_CATEGORIES)  # 19

# Friendly labels for the body-site dropdown in the UI.
LOC_LABELS = {c: c.title() for c in LOC_CATEGORIES}


def encode_meta(age: Optional[float], sex: Optional[str], loc: Optional[str]):
    """Build the 19-d metadata vector exactly as the notebook does."""
    out = []
    try:
        a = float(age) if age is not None else AGE_FILL
        if np.isnan(a):
            a = AGE_FILL
    except (TypeError, ValueError):
        a = AGE_FILL
    out.append(min(max(a, 0.0), 100.0) / 100.0)

    s = str(sex).lower().strip() if sex is not None else "unknown"
    s = s if s in SEX_CATEGORIES else "unknown"
    out += [1.0 if s == c else 0.0 for c in SEX_CATEGORIES]

    l = str(loc).lower().strip() if loc is not None else "unknown"
    l = l if l in LOC_CATEGORIES else "unknown"
    out += [1.0 if l == c else 0.0 for c in LOC_CATEGORIES]

    return np.array([out], dtype=np.float32)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Skin Cancer Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None  # lazily loaded singleton


def load_model():
    """Load the TF model once. Returns None if no model is present yet."""
    global _model
    if _model is not None:
        return _model

    import tensorflow as tf  # imported lazily so the API can boot without TF errors

    saved_model_path = MODEL_DIR / "saved_model"
    h5_path = MODEL_DIR / "model.h5"
    keras_path = MODEL_DIR / "model.keras"

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
    """True if the model has a second (metadata) input, i.e. multimodal."""
    try:
        return len(model.inputs) >= 2
    except Exception:
        return False


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes and prepare a batch tensor for the model."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file.")

    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.get("/")
def root():
    return {
        "name": "Skin Cancer Detection API",
        "model_loaded": load_model() is not None,
        "classes": CLASS_ORDER,
        "metadata_options": {
            "sex": SEX_CATEGORIES,
            "localization": LOC_CATEGORIES,
        },
        "disclaimer": (
            "Educational demo only. Not a medical device. "
            "Consult a dermatologist for any health concern."
        ),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": load_model() is not None}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    age: Optional[float] = Form(None),
    sex: Optional[str] = Form(None),
    localization: Optional[str] = Form(None),
):
    model = load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model found. Train a model in the Colab notebook and place it "
                "in backend/model/ (model.keras, model.h5, or saved_model/)."
            ),
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    batch = preprocess(image_bytes)

    # Multimodal models take [image, metadata]; single-input models take image.
    if model_expects_metadata(model):
        meta = encode_meta(age, sex, localization)
        inputs = [batch, meta]
    else:
        inputs = batch

    preds = model(inputs) if callable(model) else model.predict(inputs)
    # TFSMLayer returns a dict; normalize to an array.
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
            "malignant": CLASS_ORDER[i] in MALIGNANT,
        }
        for i in range(len(CLASS_ORDER))
    ]
    probabilities.sort(key=lambda p: p["probability"], reverse=True)

    # Combined "is this cancer?" signal: total probability across the malignant
    # classes (mel, bcc, akiec). This directly answers the benign-vs-cancer
    # question even when the single top class is benign.
    malignant_probability = float(
        sum(preds[CLASS_ORDER.index(k)] for k in MALIGNANT)
    )
    if malignant_probability >= 0.5:
        assessment = "Possibly cancerous - please see a dermatologist."
        assessment_level = "high"
    elif malignant_probability >= 0.2:
        assessment = "Uncertain - some signs of a cancerous lesion. Get it checked."
        assessment_level = "medium"
    else:
        assessment = "Likely benign (not cancer) - but this is not a diagnosis."
        assessment_level = "low"

    return {
        "prediction": {
            "key": top_key,
            "label": CLASS_LABELS[top_key],
            "confidence": float(preds[top_idx]),
            "malignant": top_key in MALIGNANT,
        },
        "cancer_assessment": {
            "is_cancer_likely": malignant_probability >= 0.5,
            "malignant_probability": malignant_probability,
            "benign_probability": 1.0 - malignant_probability,
            "level": assessment_level,
            "message": assessment,
        },
        "probabilities": probabilities,
        "disclaimer": (
            "This result is from an educational model and may be wrong. "
            "It is not a diagnosis. Please consult a qualified dermatologist."
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
