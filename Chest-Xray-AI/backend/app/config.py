"""Runtime configuration, read from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(BACKEND_DIR.parent / ".env")

# Class order == label ids used in the notebooks (Label_ClassName_Dict).
CLASS_NAMES: list[str] = ["COVID-19", "Lung-Opacity", "Normal", "Viral Pneumonia", "Tuberculosis"]

# Preprocessing constants taken from the notebook (cells 13-25).
ZOOM_FACTOR = 0.83
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = 15
INPUT_SIZE = 224

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_IMAGE_PIXELS = 50_000_000


def _path(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


MODEL_FILENAME = "final_model.h5"   # NB: not "*.weights.h5" - tf-keras would read that as the Keras-3 format
CONFIG_FILENAME = "model_config.json"
MEAN_FILENAME = "mean_image.npy"


@dataclass(frozen=True)
class Settings:
    model_dir: Path          # holds final_model.h5, model_config.json (+ mean_image.npy if needed)
    reports_dir: Path        # holds the final evaluation results written by ml/finalize.py
    frontend_dist: Path      # built website served by FastAPI (optional)
    frontend_urls: list[str]
    max_upload_bytes: int
    log_level: str


def get_settings() -> Settings:
    urls = [u.strip() for u in os.environ.get(
        "FRONTEND_URL", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000"
    ).split(",") if u.strip()]
    return Settings(
        model_dir=_path(os.environ.get("MODEL_DIR", "./models")),
        reports_dir=_path(os.environ.get("REPORTS_DIR", "../reports/final")),
        frontend_dist=_path(os.environ.get("FRONTEND_DIST", "../frontend/dist")),
        frontend_urls=urls,
        max_upload_bytes=int(float(os.environ.get("MAX_UPLOAD_SIZE_MB", "10")) * 1024 * 1024),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    )
