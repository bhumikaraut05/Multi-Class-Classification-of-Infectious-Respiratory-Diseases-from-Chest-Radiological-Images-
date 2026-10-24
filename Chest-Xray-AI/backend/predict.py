"""Standalone inference from the command line:  python predict.py path/to/xray.png"""
import sys
from pathlib import Path

from app.config import get_settings
from app.model import ModelService
from app.prediction import predict_image
from app.preprocessing import validate_image


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python predict.py <image.jpg|png>")
        return 2
    path = Path(sys.argv[1])
    data = path.read_bytes()
    settings = get_settings()
    validate_image(path.name, data, settings.max_upload_bytes)
    svc = ModelService(settings)
    svc.load()
    if not svc.loaded:
        print("Model unavailable:", svc.status_message())
        return 1
    res = predict_image(svc, data, path.name)
    print(f"AI classification: {res.prediction}  (model-estimated confidence {res.confidence:.2%})")
    for k, v in res.probabilities.items():
        print(f"  {k:<16}{v:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
