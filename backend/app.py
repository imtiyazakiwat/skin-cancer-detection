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
from datetime import datetime
from typing import Optional

# The EfficientNetV2S weights are an older Keras 2 .h5 file. Force the Keras 2
# compatibility backend (tf-keras) so it deserializes correctly under
# TensorFlow 2.16+ / Keras 3. MUST be set before TensorFlow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
# Temperature scaling the model author calibrated (T = 2.77) for reliable
# probabilities. Set to 1.0 to disable.
TEMPERATURE = float(os.getenv("TEMPERATURE", "2.77"))
MAX_FILE_MB = 10

# Model files live locally in backend/model/ and are provisioned by run.py.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_BASE_DIR, "model")
LESION_MODEL_PATH = os.path.join(MODEL_DIR, "efficientnetv2s.h5")

# --- Not-a-skin guard (zero-shot CLIP gate) -------------------------------
# A small quantized CLIP vision encoder + precomputed text-prompt embeddings
# decide whether an upload actually looks like a skin image before we run the
# lesion model. This rejects objects/food/animals/etc. without torch at
# runtime (onnxruntime only). Set SKIN_GATE=0 to disable.
GATE_ENABLED = os.getenv("SKIN_GATE", "1") != "0"
GATE_THRESHOLD = float(os.getenv("GATE_THRESHOLD", "0.5"))
GATE_NPZ = os.path.join(MODEL_DIR, "clip_gate.npz")
GATE_ONNX = os.path.join(MODEL_DIR, "clip_vision.onnx")

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

# Educational, plain-language guidance per lesion type (NOT medical advice).
CLASS_ADVICE = {
    "akiec": {
        "about": "Actinic keratoses / early intraepithelial carcinoma are rough, "
                 "scaly patches from long-term sun exposure. They are pre-cancerous "
                 "or very early cancer.",
        "action": "See a dermatologist soon. Often treated with cryotherapy, "
                  "topical creams, or a minor procedure.",
    },
    "bcc": {
        "about": "Basal cell carcinoma is the most common skin cancer. It grows "
                 "slowly and rarely spreads, but it does need treatment.",
        "action": "Book a dermatologist appointment. BCC is highly treatable, "
                  "especially when caught early.",
    },
    "bkl": {
        "about": "Benign keratosis-like lesions (e.g. seborrheic keratoses, "
                 "sun spots) are non-cancerous and very common with age.",
        "action": "Usually harmless. Mention it at a routine skin check; see a "
                  "doctor sooner if it changes.",
    },
    "df": {
        "about": "Dermatofibroma is a common benign skin nodule, often firm and "
                 "found on the legs. It is not cancer.",
        "action": "Generally no treatment needed. Get it checked if it grows, "
                  "bleeds, or becomes painful.",
    },
    "mel": {
        "about": "Melanoma is the most serious skin cancer. Early detection is "
                 "critical because it can spread.",
        "action": "See a dermatologist promptly (within days). Do not wait - early "
                  "melanoma is very treatable.",
    },
    "nv": {
        "about": "Melanocytic nevi are ordinary moles and are usually benign.",
        "action": "Monitor with the ABCDE rule. See a doctor if it changes in size, "
                  "shape, or color, or starts to itch or bleed.",
    },
    "vasc": {
        "about": "Vascular lesions (e.g. angiomas, hemangiomas) are made of blood "
                 "vessels and are almost always benign.",
        "action": "Usually harmless. Get any rapidly changing or bleeding lesion "
                  "checked.",
    },
}

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
    """Load the local lesion model (provisioned by run.py). Cached singleton."""
    global _model
    if _model is not None:
        return _model

    import tensorflow as tf  # imported lazily so the app can boot without TF

    if not os.path.exists(LESION_MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {LESION_MODEL_PATH}. "
            "Run 'python run.py' to download the model files first."
        )
    try:
        _model = tf.keras.models.load_model(LESION_MODEL_PATH)
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
    arr = (arr - 0.5) * 2.0  # EfficientNetV2S was trained on inputs in [-1, 1]
    return np.expand_dims(arr, axis=0)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale softmax probabilities (softmax(log(p) / T)).

    Equivalent to scaling the logits by 1/T, but works directly from the
    model's softmax output. Sharpens (T<1) or softens (T>1) the distribution.
    """
    if not temperature or temperature == 1.0:
        return probs
    logits = np.log(np.clip(probs.astype(np.float64), 1e-12, 1.0)) / temperature
    logits -= logits.max()
    exp = np.exp(logits)
    return (exp / exp.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Not-a-skin guard: zero-shot CLIP gate (onnxruntime only, no torch)
# ---------------------------------------------------------------------------
_gate = None          # loaded singleton
_gate_failed = False  # if assets missing/broken, fail open (skip the gate)


def load_gate():
    """Lazily load the CLIP gate (ONNX session + precomputed text embeddings)."""
    global _gate, _gate_failed
    if _gate is not None or _gate_failed or not GATE_ENABLED:
        return _gate
    try:
        import onnxruntime as ort

        data = np.load(GATE_NPZ, allow_pickle=True)
        session = ort.InferenceSession(
            GATE_ONNX, providers=["CPUExecutionProvider"]
        )
        _gate = {
            "session": session,
            "input": session.get_inputs()[0].name,
            "text_embeds": data["text_embeds"].astype(np.float32),
            "is_skin": data["is_skin"].astype(bool),
            "prompts": data["prompts"],
            "logit_scale": float(data["logit_scale"]),
            "mean": data["image_mean"].astype(np.float32),
            "std": data["image_std"].astype(np.float32),
        }
    except Exception:
        # Missing/broken assets -> don't block predictions, just skip the gate.
        _gate_failed = True
        _gate = None
    return _gate


def _clip_preprocess(image_bytes: bytes, mean, std, size: int = 224) -> np.ndarray:
    """CLIP preprocessing: resize shortest edge, center crop, normalize (NCHW)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if w <= h:
        nw, nh = size, max(size, round(h * size / w))
    else:
        nw, nh = max(size, round(w * size / h)), size
    img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - size) // 2, (nh - size) // 2
    img = img.crop((left, top, left + size, top + size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def check_is_skin(image_bytes: bytes) -> dict:
    """Decide whether the photo looks like a skin image (vs an object/etc.)."""
    gate = load_gate()
    if gate is None:
        # Gate disabled or unavailable -> allow through (fail open).
        return {"checked": False, "is_skin": True, "skin_score": None, "label": None}

    x = _clip_preprocess(image_bytes, gate["mean"], gate["std"])
    emb = gate["session"].run(None, {gate["input"]: x})[0][0].astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)   # CLIP image_embeds are un-normalized
    logits = gate["logit_scale"] * (gate["text_embeds"] @ emb)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    skin_score = float(probs[gate["is_skin"]].sum())
    top = int(np.argmax(probs))
    return {
        "checked": True,
        "is_skin": skin_score >= GATE_THRESHOLD,
        "skin_score": round(skin_score, 3),
        "label": str(gate["prompts"][top]),
    }


def build_advice(top_key: str, level: str, malignant_percent: float) -> dict:
    """Build tailored, exportable guidance from the prediction (educational)."""
    base = CLASS_ADVICE.get(top_key, {})
    if level == "high":
        headline = "See a dermatologist promptly"
        urgency = "Within a few days"
        steps = [
            "Book an appointment with a dermatologist as soon as possible.",
            "Avoid sun on the area and don't scratch or irritate it.",
            "Take clear, well-lit photos to track any changes until your visit.",
        ]
    elif level == "medium":
        headline = "Get this checked by a doctor"
        urgency = "Within 1-2 weeks"
        steps = [
            "Schedule a dermatology or GP appointment to have the lesion examined.",
            "Watch for the ABCDE warning signs listed below.",
            "Photograph the lesion now so you can compare it over time.",
        ]
    else:
        headline = "Likely benign - keep an eye on it"
        urgency = "Routine / next checkup"
        steps = [
            "No urgent action needed, but monitor the lesion monthly.",
            "Use the ABCDE rule and note any changes.",
            "Mention it at your next routine skin check.",
        ]
    return {
        "headline": headline,
        "urgency": urgency,
        "about": base.get("about", ""),
        "recommended_action": base.get("action", ""),
        "steps": steps,
        "abcde": [
            "A - Asymmetry: one half doesn't match the other.",
            "B - Border: edges are irregular, ragged, or blurred.",
            "C - Color: uneven shades of brown, black, red, white, or blue.",
            "D - Diameter: larger than 6 mm (about a pencil eraser).",
            "E - Evolving: changing in size, shape, color, or symptoms.",
        ],
        "malignant_percent": malignant_percent,
    }


def run_prediction(image_bytes: bytes, age, sex, localization) -> dict:
    """Core inference shared by the HTML form and the JSON API."""
    # --- Not-a-skin guard: reject non-skin photos before running the model. ---
    gate = check_is_skin(image_bytes)
    if gate["checked"] and not gate["is_skin"]:
        nice = (gate["label"] or "something that isn't skin").replace(
            "a photo of ", ""
        ).replace("a screenshot of ", "")
        warning = (
            f"This looks like {nice}, not a skin lesion. "
            "Please upload a clear, close-up photo of the skin lesion."
        )
        return {
            "input_check": {
                "looks_like_skin": False,
                "skin_score": gate["skin_score"],
                "detected": gate["label"],
                "warning": warning,
            },
            "prediction": None,
            "cancer_assessment": None,
            "probabilities": None,
            "disclaimer": None,
        }

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
    preds = apply_temperature(preds, TEMPERATURE)

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
        "input_check": {
            "looks_like_skin": True,
            "skin_score": gate["skin_score"],
            "detected": None,
            "warning": None,
        },
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
        "advice": build_advice(top_key, level, round(malignant_probability * 100, 1)),
        "disclaimer": (
            "This result is from an educational model and may be wrong. "
            "It is not a diagnosis. Please consult a qualified dermatologist."
        ),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def model_present() -> bool:
    """True if a model is configured (the HF model is downloaded on demand)."""
    return _model is not None or bool(HF_REPO_ID)


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
        "generated_at": datetime.now().strftime("%d %b %Y, %H:%M"),
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
    try:
        loaded = load_model() is not None
    except Exception:
        loaded = False
    return jsonify({"status": "ok", "model_loaded": loaded})


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
