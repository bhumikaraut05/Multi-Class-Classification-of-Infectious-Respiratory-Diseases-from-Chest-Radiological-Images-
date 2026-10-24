"""Shared evaluation helpers (used by evaluate.py, select_model.py and finalize.py)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .common import read_json
from .data import ImageSource, make_dataset
from .metrics import compute_metrics
from .models import build_model


def load_run(run_dir: Path):
    """Rebuild the architecture from run_config.json and load its trained weights."""
    run_dir = Path(run_dir)
    rc = read_json(run_dir / "run_config.json")
    model = build_model(rc["model"], rc["preprocessing"]["input_size"], pretrained=False)
    model.load_weights(str(run_dir / "weights.h5"))
    mean = np.load(run_dir / "mean_image.npy") if (run_dir / "mean_image.npy").exists() else None
    return model, rc, mean


def predict_probs(model, source: ImageSource, indices: np.ndarray, cfg: dict, mean, batch: int = 32):
    ds = make_dataset(source, indices, cfg, mean, batch, training=False)
    probs = model.predict(ds, verbose=0)
    return probs, source.labels[indices]


def evaluate_split(model, source, indices, cfg, mean) -> tuple[dict, np.ndarray, np.ndarray]:
    probs, y = predict_probs(model, source, indices, cfg, mean)
    return compute_metrics(y, probs), probs, y


def time_inference(model, source, cfg, mean, n: int = 20) -> float:
    """Mean milliseconds per single-image prediction (preprocessing excluded), after one warm-up call."""
    from app.preprocessing import preprocess_array
    xs = [preprocess_array(source.get(i), cfg, mean)[0][None] for i in range(min(n, len(source)))]
    model(xs[0], training=False)
    t = time.perf_counter()
    for x in xs:
        model(x, training=False)
    return (time.perf_counter() - t) / len(xs) * 1000
