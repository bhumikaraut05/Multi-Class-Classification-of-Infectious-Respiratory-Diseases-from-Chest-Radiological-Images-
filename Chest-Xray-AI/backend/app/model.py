"""Model service: loads the ONE final trained model once at startup and exposes its status."""
from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .config import CONFIG_FILENAME, MEAN_FILENAME, MODEL_FILENAME, Settings

log = logging.getLogger("cxr.model")


@dataclass
class ModelService:
    settings: Settings
    config: dict | None = None
    classifier: object | None = None       # image -> probabilities
    explainer: object | None = None        # CNN: [last_conv, probs]; ViT: [probs, attn...]
    mean_image: np.ndarray | None = None
    missing: list[str] = field(default_factory=list)
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    results: "OrderedDict[str, dict]" = field(default_factory=OrderedDict)   # recent predictions (for PDF reports)

    @property
    def loaded(self) -> bool:
        return self.classifier is not None

    @property
    def class_names(self) -> list[str]:
        from .config import CLASS_NAMES
        return (self.config or {}).get("class_names", CLASS_NAMES)

    def status_message(self) -> str:
        if self.loaded:
            return "Model loaded"
        if self.missing:
            return "Missing model files: " + ", ".join(self.missing) + ". Train and export the model first (see README)."
        return self.error or "Model not loaded"

    def remember(self, pid: str, result: dict, keep: int = 32) -> None:
        self.results[pid] = result
        while len(self.results) > keep:
            self.results.popitem(last=False)

    def load(self) -> None:
        """Load config + weights. Never raises: failures become a clear 'model unavailable' state."""
        d = self.settings.model_dir
        self.missing = [n for n in (CONFIG_FILENAME, MODEL_FILENAME) if not (d / n).is_file()]
        if self.missing:
            log.warning("Model unavailable. Missing in %s: %s", d.name, self.missing)
            return
        try:
            cfg = json.loads((d / CONFIG_FILENAME).read_text())
            mean = None
            if cfg["preprocessing"].get("mean_subtract"):
                mp = d / (cfg.get("mean_image") or MEAN_FILENAME)
                if not mp.is_file():
                    self.missing = [mp.name]
                    return
                mean = np.load(mp).astype(np.float32)
            size = int(cfg["input_size"])
            if cfg["architecture"] == "vit":
                from .vit_arch import build_vit
                clf, explainer = build_vit()
            else:
                from tensorflow.keras import Model
                from ml.models import build_model
                clf = build_model(cfg["architecture"], size, pretrained=False)
                explainer = None
            clf.load_weights(str(d / cfg.get("weights_file", MODEL_FILENAME)))
            if cfg["architecture"] != "vit":
                layer = cfg["explainability"]["layer"]
                explainer = Model(clf.inputs, [clf.get_layer(layer).output, clf.output])
            # Sanity check: a real forward pass must give a valid probability vector.
            out = np.asarray(clf(np.zeros((1, size, size, 3), np.float32), training=False))[0]
            if out.shape[0] != len(cfg["class_names"]) or not np.isfinite(out).all() or abs(out.sum() - 1) > 1e-3:
                raise ValueError("model output is not a valid probability vector")
            self.config, self.classifier, self.explainer, self.mean_image = cfg, clf, explainer, mean
            log.info("Model '%s' loaded (%s parameters)", cfg["architecture"], f"{clf.count_params():,}")
        except Exception as exc:  # noqa: BLE001 - details only in the server log
            self.classifier = self.explainer = self.mean_image = self.config = None
            self.error = "Model files were found but could not be loaded. Check the server log."
            log.exception("Failed to load model: %s", exc)

    # ------------------------------------------------------------------ info for /api/model-info
    def info(self) -> dict:
        d, rep = self.settings.model_dir, self.settings.reports_dir
        cfg = self.config
        if cfg is None and (d / CONFIG_FILENAME).is_file():
            try:
                cfg = json.loads((d / CONFIG_FILENAME).read_text())
            except Exception:  # noqa: BLE001
                cfg = None

        def read(name):
            p = rep / name
            try:
                return json.loads(p.read_text()) if p.is_file() else None
            except Exception:  # noqa: BLE001
                return None

        metrics, training = read("metrics.json"), read("training_config.json")
        from .config import CLASS_NAMES
        return {
            "model_loaded": self.loaded,
            "status_message": self.status_message(),
            "artifacts": {n: (d / n).is_file() for n in (CONFIG_FILENAME, MODEL_FILENAME)}
                         | ({cfg["mean_image"]: (d / cfg["mean_image"]).is_file()} if cfg and cfg.get("mean_image") else {}),
            "classes": (cfg or {}).get("class_names", CLASS_NAMES),
            "model": None if cfg is None else {
                "name": cfg["display_name"], "architecture": cfg["architecture"], "version": cfg["model_version"],
                "input_size": cfg["input_size"], "parameters": cfg.get("parameters"),
                "preprocessing": cfg["preprocessing"], "explainability": cfg["explainability"],
                "selection_criterion": cfg.get("selection_criterion"),
                "smoke_test_only": cfg.get("smoke_test_only", False),
            },
            "metrics": metrics,
            "training": None if training is None else {k: training.get(k) for k in (
                "hyperparameters", "optimizer", "class_weights", "seed", "epochs_run", "best_epoch", "train_seconds",
                "environment", "split_sizes", "class_counts", "pretrained_imagenet_init")},
            "history": read("history.json"),
            "model_selection": read("model_selection.json"),
        }
