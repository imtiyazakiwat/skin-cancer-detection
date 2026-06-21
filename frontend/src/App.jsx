import { useCallback, useRef, useState } from "react";

// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.js).
// In production, set VITE_API_BASE to your backend URL.
const API_BASE = import.meta.env.VITE_API_BASE || "/api";

const MAX_FILE_MB = 10;

// Must match LOC_CATEGORIES in backend/main.py.
const BODY_SITES = [
  "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
  "hand", "lower extremity", "neck", "scalp", "trunk", "unknown",
  "upper extremity",
];

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  // Patient metadata (optional, improves accuracy with the multimodal model).
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("unknown");
  const [site, setSite] = useState("unknown");
  const inputRef = useRef(null);

  const handleFiles = useCallback((files) => {
    setError(null);
    setResult(null);
    const f = files?.[0];
    if (!f) return;

    if (!f.type.startsWith("image/")) {
      setError("Please choose an image file (JPG or PNG).");
      return;
    }
    if (f.size > MAX_FILE_MB * 1024 * 1024) {
      setError(`Image is too large. Max ${MAX_FILE_MB} MB.`);
      return;
    }

    setFile(f);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(f);
    });
  }, []);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const analyze = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      if (age !== "") formData.append("age", age);
      formData.append("sex", sex);
      formData.append("localization", site);

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }, [file, age, sex, site]);

  const reset = useCallback(() => {
    setFile(null);
    setResult(null);
    setError(null);
    setAge("");
    setSex("unknown");
    setSite("unknown");
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  return (
    <div className="page">
      <header className="header">
        <h1>Skin Lesion Classifier</h1>
        <p className="subtitle">
          Upload a dermatoscopic image to get an AI screening estimate.
        </p>
      </header>

      <div className="disclaimer" role="alert">
        <strong>Educational demo only.</strong> This is not a medical device and
        does not provide a diagnosis. Always consult a qualified dermatologist
        for any health concern.
      </div>

      <main className="card">
        <div
          className={`dropzone ${dragOver ? "dropzone--over" : ""}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        >
          {previewUrl ? (
            <img src={previewUrl} alt="Selected lesion" className="preview" />
          ) : (
            <div className="dropzone__hint">
              <p className="dropzone__title">Drop an image here</p>
              <p className="dropzone__sub">or click to browse (JPG / PNG, max {MAX_FILE_MB} MB)</p>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>

        <fieldset className="meta">
          <legend className="meta__legend">
            Patient details <span className="meta__hint">(optional, improves accuracy)</span>
          </legend>
          <div className="meta__grid">
            <label className="meta__field">
              <span>Age</span>
              <input
                type="number"
                min="0"
                max="100"
                placeholder="e.g. 45"
                value={age}
                onChange={(e) => setAge(e.target.value)}
              />
            </label>
            <label className="meta__field">
              <span>Sex</span>
              <select value={sex} onChange={(e) => setSex(e.target.value)}>
                <option value="unknown">Prefer not to say</option>
                <option value="female">Female</option>
                <option value="male">Male</option>
              </select>
            </label>
            <label className="meta__field">
              <span>Body site</span>
              <select value={site} onChange={(e) => setSite(e.target.value)}>
                {BODY_SITES.map((s) => (
                  <option key={s} value={s}>
                    {s === "unknown" ? "Not sure" : s.charAt(0).toUpperCase() + s.slice(1)}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </fieldset>

        <div className="actions">
          <button
            className="btn btn--primary"
            onClick={analyze}
            disabled={!file || loading}
          >
            {loading ? "Analyzing..." : "Analyze image"}
          </button>
          <button className="btn" onClick={reset} disabled={loading && !file}>
            Reset
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        {result && <Result result={result} />}
      </main>

      <footer className="footer">
        <span>HAM10000 · CNN · FastAPI + React</span>
      </footer>
    </div>
  );
}

function Result({ result }) {
  const { prediction, probabilities, cancer_assessment: cancer } = result;
  const malignantPct = cancer ? (cancer.malignant_probability * 100).toFixed(1) : null;
  const benignPct = cancer ? (cancer.benign_probability * 100).toFixed(1) : null;

  return (
    <section className="result">
      {cancer && (
        <div className={`assessment assessment--${cancer.level}`}>
          <div className="assessment__title">
            {cancer.level === "low" ? "Likely NOT cancer" : "Possible cancer"}
          </div>
          <div className="assessment__message">{cancer.message}</div>
          <div className="assessment__split">
            <div className="assessment__metric">
              <span className="assessment__num">{malignantPct}%</span>
              <span className="assessment__cap">cancerous</span>
            </div>
            <div className="assessment__metric">
              <span className="assessment__num">{benignPct}%</span>
              <span className="assessment__cap">benign</span>
            </div>
          </div>
        </div>
      )}

      <h3 className="result__heading">Most likely lesion type</h3>
      <div
        className={`verdict ${prediction.malignant ? "verdict--alert" : "verdict--ok"}`}
      >
        <div className="verdict__label">
          {prediction.label}
          <span className={`badge ${prediction.malignant ? "badge--alert" : "badge--ok"}`}>
            {prediction.malignant ? "cancerous type" : "benign type"}
          </span>
        </div>
        <div className="verdict__confidence">
          {(prediction.confidence * 100).toFixed(1)}% confidence
        </div>
      </div>

      <h3 className="result__heading">All classes</h3>
      <ul className="bars">
        {probabilities.map((p) => (
          <li key={p.key} className="bar">
            <div className="bar__top">
              <span className="bar__label">
                {p.label}
                {p.malignant && <span className="tag-malignant">cancerous</span>}
              </span>
              <span className="bar__value">{(p.probability * 100).toFixed(1)}%</span>
            </div>
            <div className="bar__track">
              <div
                className={`bar__fill ${p.malignant ? "bar__fill--alert" : ""}`}
                style={{ width: `${Math.max(p.probability * 100, 1)}%` }}
              />
            </div>
          </li>
        ))}
      </ul>

      <p className="result__note">{result.disclaimer}</p>
    </section>
  );
}
