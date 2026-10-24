"""Paths, hyper-parameters and small helpers shared by the ML scripts."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent
import os

REPORTS_DIR = Path(os.environ.get("CXR_REPORTS_DIR", PROJECT_DIR / "reports"))   # override for smoke tests
MODELS_DIR = BACKEND_DIR / "models"

CANDIDATES = ["baseline_cnn", "vgg19", "resnet50", "custom_cnn_sa", "vit"]
DISPLAY_NAMES = {
    "baseline_cnn": "Baseline CNN",
    "vgg19": "VGG19 (ImageNet fine-tuned)",
    "resnet50": "ResNet50 (ImageNet fine-tuned)",
    "custom_cnn_sa": "Custom CNN with Spatial Attention",
    "vit": "Vision Transformer (ViT)",
}
# Per-model defaults.  Pretrained backbones use a smaller learning rate.
HPARAMS = {
    "baseline_cnn":  dict(lr=1e-3, batch=32, epochs=40),
    "vgg19":         dict(lr=1e-4, batch=32, epochs=30),
    "resnet50":      dict(lr=1e-4, batch=32, epochs=30),
    "custom_cnn_sa": dict(lr=1e-3, batch=32, epochs=60),
    "vit":           dict(lr=1e-4, batch=16, epochs=80),   # 89.7 M parameters: needs a large-memory machine/GPU
}
EARLY_STOP_PATIENCE = 8
IMBALANCE_RATIO_FOR_WEIGHTS = 1.5   # use class weights only if max/min class count exceeds this


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": __import__("os").cpu_count()}
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        info["gpus"] = [g.name for g in tf.config.list_physical_devices("GPU")]
    except Exception:  # noqa: BLE001
        pass
    for mod in ("numpy", "cv2", "sklearn"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    return info
