"""Attention-map generation for the ViT.

The notebook's own attention plotting (cell 79) slices the *heads* axis and
mixes shapes, so it is not directly reusable.  We use the same signal it
extracts (``return_attention_scores=True`` from every MultiHeadAttention
block) and aggregate it with attention rollout (Abnar & Zuidema, 2020).

The classifier pools over *all* tokens (sum pooling + global average pooling),
not just the class token, so the importance of a patch is the average rollout
attention it *receives* from all tokens.
"""
from __future__ import annotations

import cv2
import numpy as np

from .vit_arch import GRID, IMAGE_SIZE, PATCH_SIZE


def attention_rollout(block_scores: list[np.ndarray]) -> np.ndarray:
    """block_scores: list of (heads, T, T) arrays -> (GRID, GRID) patch importance in [0,1]."""
    t = block_scores[0].shape[-1]
    eye = np.eye(t, dtype=np.float64)
    rollout = eye.copy()
    for a in block_scores:
        a = a.astype(np.float64).mean(axis=0)          # average heads
        a = 0.5 * a + 0.5 * eye                         # residual connection
        a /= a.sum(axis=-1, keepdims=True)
        rollout = a @ rollout
    importance = rollout[:, 1:].mean(axis=0)            # attention received by each patch
    grid = importance.reshape(GRID, GRID)
    rng = grid.max() - grid.min()
    return (grid - grid.min()) / rng if rng > 0 else np.zeros_like(grid)


def render_attention(grid: np.ndarray, base_rgb: np.ndarray, alpha: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(heatmap_rgb, overlay_rgb)`` aligned to ``base_rgb`` (224x224)."""
    covered = GRID * PATCH_SIZE  # patches cover 220 px; the 4 px border is dropped by 'VALID'
    up = cv2.resize(grid.astype(np.float32), (covered, covered), interpolation=cv2.INTER_CUBIC)
    pad = IMAGE_SIZE - covered
    up = np.pad(up, ((0, pad), (0, pad)), mode="edge")
    up = np.clip(up, 0, 1)
    heat_bgr = cv2.applyColorMap((up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base_rgb.astype(np.uint8), 1 - alpha, heat, alpha, 0)
    return heat, overlay
