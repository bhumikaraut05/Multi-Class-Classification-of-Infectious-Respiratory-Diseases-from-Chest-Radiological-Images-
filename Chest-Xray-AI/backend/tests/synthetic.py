"""SYNTHETIC images for testing the code paths only.  They are NOT X-rays; any metric computed on them is
meaningless and must never be reported as a result of this project."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

NAMES = ["COVID-19", "Lung-Opacity", "Normal", "Viral Pneumonia", "Tuberculosis"]


def fake_xray(cls: int, rng: np.random.Generator, size: int = 256) -> np.ndarray:
    img = np.full((size, size), 40, np.float32)
    cv2.ellipse(img, (size // 3, size // 2), (size // 6, size // 3), 0, 0, 360, 120, -1)
    cv2.ellipse(img, (2 * size // 3, size // 2), (size // 6, size // 3), 0, 0, 360, 120, -1)
    for _ in range(cls * 4):                      # class-dependent "opacities"
        cv2.circle(img, tuple(rng.integers(size // 6, 5 * size // 6, 2).tolist()), int(rng.integers(size // 32 + 1, size // 12 + 2)), 200, -1)
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_folder_dataset(root: Path, per_class: int = 24, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    for c, name in enumerate(NAMES):
        d = Path(root) / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            cv2.imwrite(str(d / f"{name}_{i:03d}.png"), fake_xray(c, rng))
    return Path(root)


def png_bytes(cls: int = 2, seed: int = 1, size: int = 300, mode: str = "L") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(fake_xray(cls, np.random.default_rng(seed), size)).convert(mode).save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    print(make_folder_dataset(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 24))
