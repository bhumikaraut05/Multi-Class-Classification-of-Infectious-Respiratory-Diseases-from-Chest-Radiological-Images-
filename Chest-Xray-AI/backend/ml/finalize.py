"""Evaluate the SELECTED model on the untouched test split ONCE and export it for the website.

    python -m ml.finalize --data /path/to/dataset
Writes reports/final/* (real metrics + plots) and backend/models/{final_model.h5, model_config.json,
mean_image.npy (only if the preprocessing needs it)}.
Refuses to run twice (test-set discipline) unless --force is given.
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path

import numpy as np

from app.config import CLASS_NAMES, CONFIG_FILENAME, MEAN_FILENAME, MODEL_FILENAME
from .common import MODELS_DIR, REPORTS_DIR, read_json, write_json
from .data import load_splits, subsample
from .evaluation import evaluate_split, load_run, time_inference
from .metrics import plot_confusion, plot_curves, plot_history
from .models import EXPLAIN


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--selection", default=str(REPORTS_DIR / "selected_model.json"))
    ap.add_argument("--final-dir", default=str(REPORTS_DIR / "final"))
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="re-evaluate on the test set (discouraged)")
    a = ap.parse_args()

    final, models_dir = Path(a.final_dir), Path(a.models_dir)
    if (final / "metrics.json").exists() and not a.force:
        raise SystemExit(f"{final / 'metrics.json'} already exists. The test set should be used once; use --force to override.")
    sel = read_json(a.selection)
    run = Path(sel["run_dir"])
    model, rc, mean = load_run(run)
    test = load_splits(a.data, a.seed)["test"]
    m, probs, y = evaluate_split(model, test, subsample(test, None), rc["preprocessing"], mean)
    ms = time_inference(model, test, rc["preprocessing"], mean)

    final.mkdir(parents=True, exist_ok=True)
    weights = run / "weights.h5"
    perf = {"parameters": rc["parameters"], "weights_size_mb": round(weights.stat().st_size / 1e6, 2),
            "inference_ms_per_image_cpu_single": round(ms, 1)}
    write_json(final / "metrics.json", {**m, **perf, "split": "test", "evaluated_at": dt.datetime.now().isoformat(timespec="seconds")})
    (final / "classification_report.txt").write_text(m["classification_report"])
    plot_confusion(m["confusion_matrix"], final / "confusion_matrix.png")
    plot_curves(y, probs, final / "roc_curves.png", final / "pr_curves.png")
    hist = read_json(run / "history.json")
    write_json(final / "history.json", hist)
    plot_history(hist, final / "training_curves.png")
    write_json(final / "training_config.json", rc)
    shutil.copy(sel_path := Path(a.selection), final / "selected_model.json")
    if (sel_path.parent / "model_selection.json").exists():
        shutil.copy(sel_path.parent / "model_selection.json", final / "model_selection.json")
    write_json(final / "val_metrics.json", read_json(run / "val_metrics.json"))

    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(weights, models_dir / MODEL_FILENAME)
    if mean is not None:
        np.save(models_dir / MEAN_FILENAME, mean)
    method, layer = EXPLAIN[rc["model"]]
    write_json(models_dir / CONFIG_FILENAME, {
        "model_version": "1.0.0", "architecture": rc["model"], "display_name": rc["display_name"],
        "class_names": CLASS_NAMES, "input_size": rc["preprocessing"]["input_size"],
        "preprocessing": rc["preprocessing"], "mean_image": MEAN_FILENAME if mean is not None else None,
        "explainability": {"method": method, "layer": layer}, "weights_file": MODEL_FILENAME,
        "parameters": rc["parameters"], "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "selection_criterion": sel["criterion"], "smoke_test_only": sel.get("smoke_test_only", False),
    })
    print(m["classification_report"])
    print(f"TEST accuracy={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f}")
    print(f"Exported model to {models_dir} and reports to {final}")


if __name__ == "__main__":
    main()
