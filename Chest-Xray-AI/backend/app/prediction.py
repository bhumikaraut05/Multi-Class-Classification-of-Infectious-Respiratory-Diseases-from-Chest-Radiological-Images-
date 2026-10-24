"""Inference: upload bytes -> quality checks -> shared preprocessing -> final model -> explanation -> response."""
from __future__ import annotations

import logging
import re
import uuid

import numpy as np

from .explain import grad_cam, overlay
from .model import ModelService
from .preprocessing import last_display_stage, load_image, preprocess_array
from .quality import check_quality
from .report import build_sections, now_iso
from .schemas import PredictionResponse
from .utils import ApiError, to_data_url

log = logging.getLogger("cxr.predict")
LOW_CONFIDENCE = 0.60


def _safe_name(name: str | None) -> str:
    base = (name or "xray").replace("\\", "/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._ -]", "_", base)[:80] or "xray"


def predict_image(service: ModelService, data: bytes, filename: str | None = None) -> PredictionResponse:
    if not service.loaded:
        raise ApiError(503, "The trained model is not available. " + service.status_message())
    cfg, names = service.config, service.class_names
    original = load_image(data)
    warnings = check_quality(original)
    x_img, stages = preprocess_array(original, cfg["preprocessing"], service.mean_image)
    display = last_display_stage(stages, cfg["preprocessing"])
    x = x_img[np.newaxis, ...]

    method = cfg["explainability"]["method"]
    heat = ov = None
    note = ""
    with service.lock:
        if method == "attention_rollout":
            outputs = service.explainer(x, training=False)
            probs = np.asarray(outputs[0])[0].astype(float)
        else:
            probs = np.asarray(service.classifier(x, training=False))[0].astype(float)
        top = int(np.argmax(probs))
        try:
            if method == "grad_cam":
                heat01 = grad_cam(service.explainer, x, top)
            else:
                from .attention import attention_rollout, render_attention
                grid = attention_rollout([np.asarray(o)[0] for o in outputs[1:]])
                heat, ov = render_attention(grid, display)
                heat01 = None
            if heat01 is not None:
                heat, ov = overlay(heat01, display)
        except Exception:  # noqa: BLE001 - explanation is optional
            log.exception("Explanation failed")
            heat = ov = None
    if not np.isfinite(probs).all():
        raise ApiError(500, "Unable to analyze this image. Please try another X-ray.")
    log.info("Prediction completed")

    if method == "grad_cam":
        title, note = "Grad-CAM", ("Grad-CAM highlights image regions that most increased the model's score for the predicted "
                                   "class. It shows what influenced the model, not where disease is located.")
    else:
        title, note = "Attention map", ("The attention map shows image regions that received more attention inside the "
                                        "transformer. It is an interpretability aid, not a lesion location.")
    cfg_model = {"name": cfg["display_name"], "version": cfg["model_version"], "architecture": cfg["architecture"]}
    conf = float(probs[top])
    result = {
        "prediction_id": uuid.uuid4().hex[:12], "created_at": now_iso(), "filename": _safe_name(filename),
        "model": cfg_model, "prediction": names[top], "confidence": conf,
        "probabilities": {n: float(p) for n, p in zip(names, probs)}, "low_confidence": conf < LOW_CONFIDENCE,
        "processed_image": to_data_url(display),
        "preprocessing_steps": [{"name": n, "image": to_data_url(img)} for n, img in stages.items()],
        "explanation": {"available": heat is not None, "method": method, "title": title,
                        "heatmap": to_data_url(heat) if heat is not None else None,
                        "overlay": to_data_url(ov) if ov is not None else None, "note": note},
        "quality_warnings": warnings,
    }
    result["report"] = build_sections(result, heat is not None, note)
    service.remember(result["prediction_id"], result)
    return PredictionResponse(**result)
