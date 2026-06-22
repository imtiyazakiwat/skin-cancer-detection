# Skin Lesion Classifier

An educational skin-lesion classification demo. Train a model in Google Colab,
then run a single **Flask** app that serves a plain HTML/CSS page where you
upload an image and see a live prediction. No Node.js, React, or build step.

> **Disclaimer:** This is an educational/screening-aid demo, **not** a medical
> device. It must not be used for diagnosis. Always consult a qualified
> dermatologist for any health concern.

## Architecture

```
Google Colab (train)  ->  model.keras  ->  Flask app (HTML/CSS UI + /predict)
```

- **notebook/** - Colab notebook: HAM10000 + SqueezeNet (trained from scratch).
- **backend/**  - Flask app (`app.py`) that loads the model and serves the UI.
  - `templates/index.html` - the upload form and results page (server-rendered).
  - `static/styles.css`    - styling.
  - `model/model.keras`    - the trained model.

The model predicts 7 HAM10000 classes:
`akiec, bcc, bkl, df, mel, nv, vasc` (alphabetical order, kept in sync between
the notebook's `class_indices` and `CLASS_ORDER` in `backend/app.py`).

## Quick start (any OS)

The easiest way - one command that checks Python, creates the virtualenv,
installs requirements, and starts the app:

```bash
python run.py
```

Then open http://localhost:8000.

Options:

```bash
python run.py --setup-only   # install everything but don't start the server
python run.py --recreate     # rebuild backend/.venv from scratch
python run.py --port 8001    # use a different port
```

### Platform launchers (optional)

- **Windows:** double-click `start.bat`, or run `.\start.ps1`.
- **macOS / Linux:** run `./start.sh`.

Both do the same thing as `python run.py`.

> **Windows note:** if installing TensorFlow fails with a
> `No such file or directory` error, it's the Windows 260-character path limit.
> Either move the project to a short path like `C:\scd`, or run
> `python run.py --enable-long-paths` in an **admin** PowerShell, then
> `python run.py --recreate`.

## Manual setup

```bash
cd backend
python -m venv .venv
# Windows:  .\.venv\Scripts\activate
# mac/linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:8000.

- `GET /health` reports whether a model is loaded.
- `POST /api/predict` is a JSON API (multipart `file` + optional `age`, `sex`,
  `localization`) if you want to call it programmatically.
- If no model is present, the page shows a notice; drop a `model.keras` into
  `backend/model/` and restart.

## 1. Train the model (Google Colab)

1. Open `notebook/train_skin_cancer.ipynb` in Google Colab.
2. Set **Runtime -> Change runtime type -> GPU**.
3. Run the cells. You'll be asked to upload your `kaggle.json` token
   (Kaggle -> Account -> Create New API Token) to download HAM10000.
4. At the end it downloads `skin_model_export.zip`.
5. Unzip it and copy **`model.keras`** into `backend/model/`.

> Tip: check the `class_indices` printed in the notebook matches `CLASS_ORDER`
> in `backend/app.py`. They should both be alphabetical.

## Configuration

| Variable | Purpose |
| --- | --- |
| `MODEL_DIR` | Folder holding the model (default `backend/model`). |
| `IMG_SIZE`  | Fallback input size if it can't be read from the model (default `224`). |
| `PORT`      | Port for the Flask app (default `8000`). |
| `FLASK_DEBUG` | Set to `0` to disable debug/auto-reload. |

## Deployment notes

- Run behind a production WSGI server (e.g. `gunicorn app:app` on Linux, or
  `waitress-serve --port 8000 app:app` on Windows) instead of `python app.py`.
- Host on Render, Railway, or Hugging Face Spaces (Docker). Bake the model into
  the image or download it at startup.

## Responsible use

HAM10000 is imbalanced (the `nv` class dominates), so watch per-class recall,
especially for melanoma (`mel`), not just overall accuracy. The model can be
confidently wrong. Keep the disclaimer visible wherever this is shown.
