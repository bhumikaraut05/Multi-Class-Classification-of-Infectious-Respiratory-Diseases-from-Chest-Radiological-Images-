"""Small helpers: image encoding and API errors."""
from __future__ import annotations

import base64

import cv2
import numpy as np


class ApiError(Exception):
    """Error with an HTTP status and a message that is safe to show to users."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def to_data_url(rgb: np.ndarray) -> str:
    """Encode an RGB uint8 image as a PNG data URL."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ApiError(500, "Unable to analyze this image. Please try another X-ray.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
