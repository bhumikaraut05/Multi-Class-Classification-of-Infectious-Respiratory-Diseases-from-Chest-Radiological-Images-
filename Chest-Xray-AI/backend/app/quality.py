"""Simple, transparent image-quality heuristics (warnings only; they never change the prediction)."""
from __future__ import annotations

import cv2
import numpy as np

from .utils import ApiError


def check_quality(rgb: np.ndarray) -> list[dict]:
    """Return warnings for the ORIGINAL decoded image.  Raises ApiError for blank / unusable images."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    std, mean = float(gray.std()), float(gray.mean())
    if std < 2.0:
        raise ApiError(422, "The image appears blank or uniform, so it cannot be analysed. Please upload a chest X-ray.")
    out = []
    if min(h, w) < 224:
        out.append({"code": "low_resolution", "message": f"Low resolution ({w}x{h}); the image was enlarged to 224x224, which can reduce reliability."})
    if std < 25:
        out.append({"code": "low_contrast", "message": "The image has very low contrast, which may reduce reliability."})
    if mean < 25 or mean > 230:
        out.append({"code": "exposure", "message": "The image is very dark or very bright, which may reduce reliability."})
    colourfulness = float(np.abs(rgb[..., 0].astype(np.int16) - rgb[..., 1]).mean() + np.abs(rgb[..., 1].astype(np.int16) - rgb[..., 2]).mean())
    if colourfulness > 12:
        out.append({"code": "colour_image", "message": "The image is in colour. Chest X-rays are normally greyscale, so this may not be a radiograph."})
    ar = w / h
    if ar > 1.5 or ar < 1 / 1.5:
        out.append({"code": "aspect_ratio", "message": f"Unusual shape ({w}x{h}); the image was resized to a square, which distorts it."})
    return out
