"""Evaluate a trained run on a split and save metrics + plots next to it.

    python -m ml.evaluate --data /path/to/dataset --run reports/runs/custom_cnn_sa__clahe --split val
Use --split test ONLY for the final selected model (ml.finalize does this for you, once).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import write_json
from .data import load_splits, subsample
from .evaluation import evaluate_split, load_run
from .metrics import plot_confusion, plot_curves


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"]); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run = Path(a.run)
    model, rc, mean = load_run(run)
    src = load_splits(a.data, a.seed)[a.split]
    m, probs, y = evaluate_split(model, src, subsample(src, None), rc["preprocessing"], mean)
    write_json(run / f"{a.split}_metrics.json", m)
    (run / f"{a.split}_classification_report.txt").write_text(m["classification_report"])
    plot_confusion(m["confusion_matrix"], run / f"{a.split}_confusion_matrix.png")
    plot_curves(y, probs, run / f"{a.split}_roc_curves.png", run / f"{a.split}_pr_curves.png")
    print(m["classification_report"])
    print(f"accuracy={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} macro-AUC={m['macro_auc_ovr']}")


if __name__ == "__main__":
    main()
