"""Train ONE candidate on the train split, early-stopping on the validation split.

    python -m ml.train --data /path/to/dataset --model custom_cnn_sa
The test split is never used here.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import common as C
from .common import CANDIDATES, DISPLAY_NAMES, HPARAMS, REPORTS_DIR, write_json
from .data import class_counts, class_weights_if_needed, load_splits, make_dataset, subsample
from .evaluation import evaluate_split
from .metrics import plot_confusion, plot_history


def compute_mean_image(source, indices, cfg, cap: int = 5000, seed: int = 0) -> np.ndarray:
    """True pixel-wise mean of the (pre-mean-subtraction) training images."""
    from app.preprocessing import preprocess_array
    idx = np.random.default_rng(seed).choice(indices, min(cap, len(indices)), replace=False)
    c = dict(cfg, mean_subtract=False)
    acc = np.zeros((cfg["input_size"],) * 2 + (3,), np.float64)
    for i in idx:
        acc += preprocess_array(source.get(int(i)), c)[0]
    return (acc / len(idx)).astype(np.float32)


def run_training(data, model_name, preset="clahe", out_root=None, epochs=None, batch=None, lr=None,
                 pretrained=None, seed=42, limit=None) -> Path:
    import tensorflow as tf
    from app.preprocessing import PRESETS
    from .models import build_model

    if model_name not in CANDIDATES:
        raise SystemExit(f"--model must be one of {CANDIDATES}")
    tf.keras.utils.set_random_seed(seed)
    hp = dict(HPARAMS[model_name])
    hp.update({k: v for k, v in dict(epochs=epochs, batch=batch, lr=lr).items() if v})
    cfg = dict(PRESETS[preset])
    pretrained = model_name in ("vgg19", "resnet50") if pretrained is None else pretrained
    run_dir = Path(out_root or REPORTS_DIR / "runs") / f"{model_name}__{preset}"
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(data, seed)
    tr_idx, va_idx = subsample(splits["train"], limit), subsample(splits["val"], limit)
    mean = compute_mean_image(splits["train"], tr_idx, cfg) if cfg["mean_subtract"] else None
    if mean is not None:
        np.save(run_dir / "mean_image.npy", mean)

    cw, cw_info = class_weights_if_needed(splits["train"].labels[tr_idx], C.IMBALANCE_RATIO_FOR_WEIGHTS)
    train_ds = make_dataset(splits["train"], tr_idx, cfg, mean, hp["batch"], True, seed)
    val_ds = make_dataset(splits["val"], va_idx, cfg, mean, hp["batch"], False)

    model = build_model(model_name, cfg["input_size"], pretrained=pretrained)
    opt = tf.keras.optimizers.Adam(hp["lr"])
    if model_name == "vit":
        try:
            opt = tf.keras.optimizers.AdamW(hp["lr"], weight_decay=1e-3)
        except Exception:  # noqa: BLE001
            pass
    model.compile(opt, loss="categorical_crossentropy", metrics=["accuracy"])
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE,
                                            restore_best_weights=True, verbose=1),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)]
    t0 = time.time()
    h = model.fit(train_ds, validation_data=val_ds, epochs=hp["epochs"], class_weight=cw, callbacks=cbs, verbose=2)
    seconds = time.time() - t0
    hist = {k: [float(x) for x in v] for k, v in h.history.items()}
    model.save_weights(str(run_dir / "weights.h5"))
    write_json(run_dir / "history.json", hist)
    plot_history(hist, run_dir / "training_curves.png")

    val_metrics, _, _ = evaluate_split(model, splits["val"], va_idx, cfg, mean)
    write_json(run_dir / "val_metrics.json", val_metrics)
    plot_confusion(val_metrics["confusion_matrix"], run_dir / "val_confusion_matrix.png")
    write_json(run_dir / "run_config.json", {
        "model": model_name, "display_name": DISPLAY_NAMES[model_name], "preset": preset, "preprocessing": cfg,
        "hyperparameters": hp, "optimizer": type(opt).__name__, "pretrained_imagenet_init": pretrained,
        "class_weights": cw_info, "seed": seed, "epochs_run": len(hist["loss"]),
        "best_epoch": int(np.argmin(hist["val_loss"])) + 1, "train_seconds": round(seconds, 1),
        "subsampled_for_smoke_test": bool(limit), "environment": C.environment_info(),
        "split_sizes": {s: len(src) for s, src in splits.items()},
        "class_counts": {s: class_counts(src.labels) for s, src in splits.items()},
        "parameters": int(model.count_params()),
    })
    print(f"[{model_name}] val acc={val_metrics['accuracy']:.4f} macro-F1={val_metrics['macro_f1']:.4f} -> {run_dir}")
    return run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, choices=CANDIDATES)
    ap.add_argument("--preprocessing", default="clahe", choices=["plain", "clahe", "legacy"])
    ap.add_argument("--out", default=None, help="parent directory for the run folder (default reports/runs)")
    ap.add_argument("--epochs", type=int); ap.add_argument("--batch", type=int); ap.add_argument("--lr", type=float)
    ap.add_argument("--no-pretrained", action="store_true", help="do not initialise VGG19/ResNet50 from ImageNet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="SMOKE TESTS ONLY: use a random subset per split")
    a = ap.parse_args()
    run_training(a.data, a.model, a.preprocessing, a.out, a.epochs, a.batch, a.lr,
                 False if a.no_pretrained else None, a.seed, a.limit)


if __name__ == "__main__":
    main()
