# Skin Lesion Classifier

An educational skin-lesion classification demo. Train a model in Google Colab,
serve it with a FastAPI backend, and run a React frontend that lets you upload
an image and see a live prediction.

> **Disclaimer:** This is an educational/screening-aid demo, **not** a medical
> device. It must not be used for diagnosis. Always consult a qualified
> dermatologist for any health concern.

## Architecture

```
Google Colab (train)  ->  model.keras  ->  FastAPI /predict  <-  React (upload)```

- **notebook/** - Colab notebook: HAM10000 + SqueezeNet (trained from scratch).
- **backend/**  - FastAPI app exposing `/predict`.
- **frontend/** - Vite + React app with image upload.

The model predicts 7 HAM10000 classes:
`akiec, bcc, bkl, df, mel, nv, vasc` (alphabetical order, kept in sync between
the notebook's `class_indices` and `CLASS_ORDER` in `backend/main.py`).

## 1. Train the model (Google Colab)

1. Open `notebook/train_skin_cancer.ipynb` in Google Colab.
2. Set **Runtime -> Change runtime type -> GPU**.
3. Run the cells. You'll be asked to upload your `kaggle.json` token
   (Kaggle -> Account -> Create New API Token) to download HAM10000.
4. At the end it downloads `skin_model_export.zip`.
5. Unzip it and copy **`model.keras`** into `backend/model/`.

> Tip: check the `class_indices` printed in the notebook matches `CLASS_ORDER`
> in `backend/main.py`. They should both be alphabetical.

## 2. Run the backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

- API: http://localhost:8000  (interactive docs at `/docs`)
- `GET /health` reports whether a model is loaded.
- If no model is present, `/predict` returns 503 with instructions.

## 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The dev server proxies `/api` to the backend on
port 8000, so no extra config is needed locally.

Upload a dermatoscopic image and click **Analyze image** to see the predicted
class, a confidence score, and per-class probabilities.

## Configuration

| Where | Variable | Purpose |
| --- | --- | --- |
| backend | `MODEL_DIR` | Folder holding the model (default `backend/model`). |
| backend | `IMG_SIZE` | Input size, must match training (default `224`). |
| frontend | `VITE_API_BASE` | Backend URL in production (see `.env.example`). |

## Deployment notes

- **Backend:** Hugging Face Spaces (Docker) or Render work well. Bake the model
  into the image or download it at startup.
- **Frontend:** any static host (Netlify, Vercel, GitHub Pages). Set
  `VITE_API_BASE` to your deployed backend URL before `npm run build`.
- Tighten CORS `allow_origins` in `backend/main.py` to your frontend origin
  before going public.

## Responsible use

HAM10000 is imbalanced (the `nv` class dominates), so watch per-class recall,
especially for melanoma (`mel`), not just overall accuracy. The model can be
confidently wrong. Keep the disclaimer visible wherever this is shown.
