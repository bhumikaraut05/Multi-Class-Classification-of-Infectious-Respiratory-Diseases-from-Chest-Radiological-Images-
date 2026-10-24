# CXR-AI — five-class chest X-ray classification (academic research prototype)

Classes: **COVID-19, Lung Opacity, Normal, Viral Pneumonia, Tuberculosis**. One model is trained, selected on
*validation* performance, evaluated once on the *test* split, and served by a local FastAPI + React website
(prediction, probabilities, Grad-CAM/attention explanation, downloadable PDF report). No cloud, API keys or database.

> **Status of this delivery.** The code, website, tests and launcher are complete and verified (see *Verification*).
> **No model has been trained or selected yet**: the dataset and weights were not in the ZIP and the build environment
> had no dataset access/GPU. Until you run the training steps below, the site starts but shows "Model unavailable" and
> predictions are disabled. No metric anywhere in this repo is pre-filled.

## 1. Run the website (one command)
Requires Python 3.10–3.12 (3.13 works with current TensorFlow). Node.js is **not** needed (the built site is included).
```
Windows:      start.bat --setup      (first time only: creates .venv, installs backend/requirements.txt)
              start.bat
Linux/macOS:  ./start.sh --setup ; ./start.sh
```
The launcher checks Python packages, model files and ports, starts the backend, waits for `/api/health`, then prints
`READY` and opens http://127.0.0.1:8000. `Ctrl+C` or `stop.bat` / `./stop.sh` shuts everything down.
Developer mode (Vite hot reload, needs Node 18+): `python run.py --dev` → http://127.0.0.1:5173.
Docker (untested here): `docker compose up --build` → http://localhost:8000.

## 2. Get the dataset
Download the five-class chest X-ray dataset (≈32,687 images) from Harvard Dataverse (search "chest X-ray COVID-19,
Lung Opacity, Normal, Viral Pneumonia, Tuberculosis"; the original project used `Dataset5_raw_{train,val,test}.h5`).
Accepted layouts for `--data`:
* a folder with `*train*.h5`, `*val*.h5`, `*test*.h5` (datasets `image`, `image_label`, `image_name`) — official splits respected;
* `<data>/{train,val,test}/<Class>/*.png` — official splits respected;
* `<data>/<Class>/*.png` — stratified 70/15/15 split, identical files kept together, saved to `reports/splits.json`.

Class folder names are matched loosely (`COVID-19`, `Lung-Opacity`, `Normal`, `Viral Pneumonia`, `Tuberculosis`).
Audit first: `cd backend && python -m ml.prepare_dataset --data PATH` → `reports/dataset_report.json`.

## 3. Train the model (run from `backend/`, after `pip install -r requirements.txt`)
```
python -m ml.prepare_dataset --data PATH            # counts, imbalance, corrupt files, duplicates/leakage
python -m ml.select_model    --data PATH            # trains Baseline CNN, VGG19, ResNet50, Custom CNN+Spatial Attention, ViT
                                                    # on the SAME split; picks ONE by validation macro-F1
python -m ml.finalize        --data PATH            # tests the selected model ONCE; exports to backend/models/
python run.py (project root)                         # website now serves the real model
```
* Hardware: a GPU is strongly advised (use `tensorflow` instead of `tensorflow-cpu`). The ViT (89.7 M parameters) needs
  a large-memory machine: it was killed by out-of-memory on a 7 GB CPU box; each candidate runs in its own process so a
  failure is recorded and the others continue. Exclude it with `--models baseline_cnn vgg19 resnet50 custom_cnn_sa`.
* VGG19/ResNet50 start from ImageNet weights (downloaded by Keras on first use; add `--no-pretrained` to avoid this).
* Single model only: `python -m ml.train --data PATH --model custom_cnn_sa [--epochs N --preprocessing plain|clahe|legacy]`,
  `python -m ml.evaluate --data PATH --run reports/runs/<run> --split val`.
* Preprocessing presets (`--preprocessing`, default `clahe`) are applied by one shared function in training, evaluation
  and the website; try `plain`/`legacy` to ablate steps (re-run `select_model` per preset; compare validation scores).
* Augmentation (small rotation/scale/shift/brightness, no flips) is training-only; early stopping + LR reduction on val loss;
  class weights only when max/min class ratio > 1.5. Seed 42.

Outputs: `reports/runs/*` (per-candidate), `reports/model_selection.{json,md}`, `reports/final/` (metrics.json,
classification_report.txt, confusion_matrix/roc/pr/training-curve PNGs, config), and the website artifacts
`backend/models/final_model.h5`, `model_config.json` (+ `mean_image.npy` only for the `legacy` preset).
The website's *Model information* page renders these files; absent files → "no results yet".

## 4. Tests
```
cd backend  && python -m pytest -q            # 15 tests: load, predict, schema/probabilities, invalid uploads,
                                              # missing/corrupt model, PDF content, all 5 architectures
cd frontend && npm install && npm test        # 6 tests: validation, disabled state, real response displayed, report download
cd frontend && npm run build                  # rebuild frontend/dist after editing the UI
```
Backend tests use a throw-away *randomly initialised* network created inside the test run to exercise the plumbing; it
is never shipped or used by the app. `backend/tests/synthetic.py` makes fake images for smoke tests only.

## 5. Layout
```
run.py start.* stop.*     launcher            backend/app     FastAPI service (config-driven single model)
backend/ml                training/eval code  backend/models  exported model (empty until you train)
frontend/                 React + Vite        reports/        evaluation outputs (empty until you train)
docs/API.md docs/AUDIT.md docs/ORIGINAL_PROJECT_ANALYSIS.md
```
Privacy: uploads are decoded in memory, never written to disk or logged; recent results are kept in server memory
(max 32) only to build PDFs. Medical use is out of scope: outputs are AI classifications, not diagnoses.
