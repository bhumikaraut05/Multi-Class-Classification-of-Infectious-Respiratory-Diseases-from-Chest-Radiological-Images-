# Audit of the supplied ZIP

**Present:** React/Vite frontend, FastAPI backend, ViT architecture re-implementation (`vit_arch.py`, 89,710,422 parameters, verified by test), preprocessing port, tests, Docker files.

**Missing (must be supplied/produced by you):** the dataset (Harvard Dataverse 5-class set), the original 7 notebooks, `VitB16_DownSized.h5`, `mean_image.npy`, any trained weights or saved metrics. Consequently **no model could be evaluated, compared or selected in the build environment** (no dataset access, CPU-only, 7 GB RAM) and none of the previously reported numbers (ViT ≈78.7 %, Custom CNN 0.97 vs 0.92 in the report) could be reproduced or verified.

**Issues found and how they were handled**
| Issue | Resolution |
|---|---|
| Only the ViT was supported; no training/eval code | `backend/ml/` – train, evaluate, select_model, finalize, dataset audit for 5 candidates |
| "Mean image" computed by summing over rows (per-column mean) | Replaced by a true pixel-wise mean, computed from the training split and saved when the preprocessing preset needs it; the old quirk only mattered for the lost ViT weights |
| Attention pooling `Dense(1, softmax)` is constant 1.0 | Kept in `vit_arch.py` only for compatibility with any old weights; new training uses the same code, noted as a known quirk |
| Preprocessing steps never ablated | Three presets (`plain`, `clahe`, `legacy`) selectable via `--preprocessing`; the preset used is stored with the model and used identically at inference |
| Possible leakage / imbalance / duplicates | `ml.prepare_dataset` reports counts, imbalance, corrupt files, within/cross-split duplicates; folder-based splits keep byte-identical files together; class weights only if imbalance > 1.5× |
| Patient IDs absent | Patient-level leakage cannot be excluded; stated in UI and docs |
| Baseline CNN and Custom CNN + Spatial Attention code not in ZIP | Re-implemented from the description; **not** guaranteed identical to the originals |
| Test-set discipline | Selection uses validation only; `ml.finalize` evaluates test once and refuses to repeat without `--force` |
