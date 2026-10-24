"""Preprocessing shared by training (ml/), evaluation and the website.

Every step is switched on/off by a small config dict (see PRESETS) that is saved
next to the trained weights in ``models/model_config.json``.  Training and the
API both call :func:`preprocess_array`, so they cannot drift apart.

Steps (all ported from the original notebooks): zoom_into_chest_area, CLAHE per
channel, sharpening kernel, /255, optional mean-image subtraction.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import (ALLOWED_EXTENSIONS, ALLOWED_FORMATS, CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE,
                     INPUT_SIZE, MAX_IMAGE_PIXELS, ZOOM_FACTOR)
from .utils import ApiError

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# "Basic Sharpening Kernel" from the notebook (the one used in the pipeline).
SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])


MIN_SIDE = 64


def validate_image(filename: str | None, data: bytes, max_bytes: int) -> None:
    """Raise ApiError for empty, oversized, wrong-type or unreadable uploads."""
    if not data:
        raise ApiError(400, "Please upload an X-ray image.")
    if len(data) > max_bytes:
        raise ApiError(413, f"Image is too large (max {max_bytes // (1024 * 1024)} MB).")
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ApiError(415, "Please upload a JPG, JPEG, or PNG image.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ALLOWED_FORMATS:
                raise ApiError(415, "Please upload a JPG, JPEG, or PNG image.")
            if min(im.size) < MIN_SIDE:
                raise ApiError(422, f"Image is too small (minimum {MIN_SIDE}x{MIN_SIDE} pixels).")
            im.verify()
    except ApiError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ApiError(422, "The uploaded image could not be processed.")


def load_image(data: bytes) -> np.ndarray:
    """Decode bytes to an RGB uint8 array (grayscale is replicated to 3 channels)."""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ApiError(422, "The uploaded image could not be processed.")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def resize_image(img: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def apply_zoom(img: np.ndarray, zoom_factor: float = ZOOM_FACTOR) -> np.ndarray:
    """Port of ``zoom_into_chest_area``: centre-crop then resize back."""
    img = img.astype("uint8")
    cx, cy = img.shape[1] // 2, img.shape[0] // 2
    w, h = img.shape[1] * zoom_factor, img.shape[0] * zoom_factor
    x1, x2 = int(cx - w // 2), int(cx + w // 2)
    y1, y2 = int(cy - h // 2), int(cy + h // 2)
    roi = img[y1:y2, x1:x2]
    return cv2.resize(roi, (img.shape[1], img.shape[0]))


def apply_clahe(img: np.ndarray, grid_size: int = CLAHE_GRID_SIZE,
                clip_limit: float = CLAHE_CLIP_LIMIT) -> np.ndarray:
    """Port of ``apply_clahe``: CLAHE applied independently to each channel."""
    img = img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(grid_size), int(grid_size)))
    out = np.zeros_like(img)
    for c in range(3):
        out[..., c] = clahe.apply(img[..., c])
    return out


def apply_sharpening(img: np.ndarray, kernel: np.ndarray = SHARPEN_KERNEL) -> np.ndarray:
    """Port of ``sharpen_image`` (cv2.filter2D on uint8)."""
    return cv2.filter2D(img.astype(np.uint8), -1, np.array(kernel))


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Scale uint8 pixels to [0, 1] (notebook ``scaling``)."""
    return img.astype(np.float32) / 255.0


def zero_center_image(img: np.ndarray, mean_image: np.ndarray) -> np.ndarray:
    """Subtract the training-set mean image (notebook ``zero_center``)."""
    return (img - mean_image).astype(np.float32)


# Named preprocessing settings.  "legacy" is the original notebook pipeline.
PRESETS: dict[str, dict] = {
    "plain":  {"input_size": INPUT_SIZE, "zoom": None, "clahe": False, "sharpen": False, "mean_subtract": False},
    "clahe":  {"input_size": INPUT_SIZE, "zoom": None, "clahe": True,  "sharpen": False, "mean_subtract": False},
    "legacy": {"input_size": INPUT_SIZE, "zoom": ZOOM_FACTOR, "clahe": True, "sharpen": True, "mean_subtract": True},
}
DEFAULT_PRESET = "clahe"


def preprocess_array(rgb: np.ndarray, cfg: dict, mean_image: np.ndarray | None = None
                     ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """RGB uint8 image (any size) -> (float32 model input, display stages).

    Deterministic; used identically for train / val / test / website.
    """
    size = int(cfg.get("input_size", INPUT_SIZE))
    img = resize_image(rgb, size)
    stages = {"Original (resized %dx%d)" % (size, size): img}
    if cfg.get("zoom"):
        img = apply_zoom(img, float(cfg["zoom"]))
        stages["Zoom (%.2f)" % cfg["zoom"]] = img
    if cfg.get("clahe"):
        img = apply_clahe(img)
        stages["CLAHE"] = img
    if cfg.get("sharpen"):
        img = apply_sharpening(img)
        stages["Sharpening"] = img
    out = normalize_image(img)
    if cfg.get("mean_subtract"):
        if mean_image is None:
            raise ApiError(503, "The model needs mean_image.npy, which is missing.")
        out = zero_center_image(out, mean_image)
        rng = float(out.max() - out.min()) or 1.0
        stages["Zero-centred (model input, rescaled for display)"] = ((out - out.min()) / rng * 255).astype(np.uint8)
    return out, stages


def last_display_stage(stages: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    """The uint8 image that best represents what the model sees (before mean subtraction)."""
    keys = [k for k in stages if not k.startswith("Zero-centred")]
    return stages[keys[-1]]


@dataclass
class PreprocessResult:
    model_input: np.ndarray
    stages: dict[str, np.ndarray]
    display: np.ndarray
    original_shape: tuple[int, int]


def preprocess_xray(data: bytes, cfg: dict, mean_image: np.ndarray | None = None) -> PreprocessResult:
    """Full pipeline from raw upload bytes to model input."""
    original = load_image(data)
    x, stages = preprocess_array(original, cfg, mean_image)
    return PreprocessResult(x, stages, last_display_stage(stages, cfg), original.shape[:2])
