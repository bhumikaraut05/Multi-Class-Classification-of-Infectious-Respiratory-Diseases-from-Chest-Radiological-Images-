"""Explanations for the single final model: Grad-CAM (CNNs) or attention rollout (ViT)."""
from __future__ import annotations

import cv2
import numpy as np


def grad_cam(explainer, x: np.ndarray, class_index: int) -> np.ndarray:
    """Grad-CAM heat-map in [0,1] at the model's input resolution."""
    import tensorflow as tf
    with tf.GradientTape() as tape:
        conv, probs = explainer(x, training=False)
        score = probs[:, class_index]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))[0].numpy()
    cam = cv2.resize(cam, (x.shape[2], x.shape[1]), interpolation=cv2.INTER_CUBIC)
    rng = cam.max() - cam.min()
    return np.clip((cam - cam.min()) / rng, 0, 1) if rng > 1e-8 else np.zeros_like(cam)


def overlay(heat01: np.ndarray, base_rgb: np.ndarray, alpha: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    heat = cv2.cvtColor(cv2.applyColorMap((heat01 * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    base = base_rgb.astype(np.uint8)
    if base.shape[:2] != heat.shape[:2]:
        base = cv2.resize(base, (heat.shape[1], heat.shape[0]))
    return heat, cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)
