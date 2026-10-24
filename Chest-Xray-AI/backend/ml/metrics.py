"""Metrics and plots computed from real predictions only."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, auc, average_precision_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score, precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

from app.config import CLASS_NAMES  # noqa: E402


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_pred = probs.argmax(1)
    labels = list(range(len(CLASS_NAMES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = [c for c in labels if (y_true == c).any() and (y_true != c).any()]
    onehot = np.eye(len(CLASS_NAMES))[y_true]
    roc, ap = {}, {}
    for c in present:
        roc[CLASS_NAMES[c]] = float(roc_auc_score(onehot[:, c], probs[:, c]))
        ap[CLASS_NAMES[c]] = float(average_precision_score(onehot[:, c], probs[:, c]))
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_auc_ovr": float(np.mean(list(roc.values()))) if roc else None,
        "macro_auprc": float(np.mean(list(ap.values()))) if ap else None,
        "per_class": {CLASS_NAMES[c]: {"precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]),
                                       "support": int(s[c]), "roc_auc": roc.get(CLASS_NAMES[c]),
                                       "auprc": ap.get(CLASS_NAMES[c])} for c in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": CLASS_NAMES,
        "classification_report": classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
                                                       zero_division=0),
    }


def plot_confusion(cm, path):
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > .5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (counts, row %)")
    fig.colorbar(im, ax=ax, fraction=.046); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_curves(y_true, probs, roc_path, pr_path):
    onehot = np.eye(len(CLASS_NAMES))[y_true]
    for kind, path in (("roc", roc_path), ("pr", pr_path)):
        fig, ax = plt.subplots(figsize=(6, 5))
        for c, name in enumerate(CLASS_NAMES):
            if not ((y_true == c).any() and (y_true != c).any()):
                continue
            if kind == "roc":
                fpr, tpr, _ = roc_curve(onehot[:, c], probs[:, c])
                ax.plot(fpr, tpr, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
            else:
                pr, rc, _ = precision_recall_curve(onehot[:, c], probs[:, c])
                ax.plot(rc, pr, label=f"{name} (AP {average_precision_score(onehot[:, c], probs[:, c]):.3f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_xlabel("False positive rate" if kind == "roc" else "Recall")
        ax.set_ylabel("True positive rate" if kind == "roc" else "Precision")
        ax.set_title("ROC curves (one-vs-rest)" if kind == "roc" else "Precision-recall curves (one-vs-rest)")
        ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_history(hist: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(hist.get(key, []), label="train")
        ax.plot(hist.get("val_" + key, []), label="validation")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
