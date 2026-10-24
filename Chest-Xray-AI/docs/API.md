# API (FastAPI, base URL `http://127.0.0.1:8000`, interactive docs at `/docs`)

| Method & path | Purpose |
|---|---|
| `GET /api/health` | `{status, model_loaded, detail}`. `model_loaded=false` + actionable `detail` if artifacts are missing/corrupt. |
| `GET /api/model-info` | Selected algorithm, classes, artifact status, **real** saved metrics/history/training config (`null` if not evaluated yet). |
| `POST /api/predict` | multipart field `file` (JPG/JPEG/PNG, ≤10 MB, ≥64×64 px). Returns prediction, confidence, all class probabilities, preprocessing images, Grad-CAM / attention explanation, quality warnings, report sections, `prediction_id`. |
| `POST /api/report` | JSON `{"prediction_id": "..."}` → `application/pdf`, built from the stored real prediction (kept in memory, last 32). |
| `GET /api/evaluation-plot/{confusion_matrix\|roc_curves\|pr_curves\|training_curves}` | PNG written by `ml.finalize`. |

Errors are `{"error": "<safe message>"}`: 400 no file, 404 unknown report, 413 too large, 415 wrong type, 422 undecodable/too small/blank image, 503 model unavailable. No stack traces or paths are returned.
