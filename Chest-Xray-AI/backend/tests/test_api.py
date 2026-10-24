import math

import numpy as np

from app.config import CLASS_NAMES
from tests.synthetic import png_bytes


def post(client, data, name="x.png", ctype="image/png"):
    return client.post("/api/predict", files={"file": (name, data, ctype)})


def test_model_loads_and_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "healthy" and h["model_loaded"] is True
    info = client.get("/api/model-info").json()
    assert info["model_loaded"] and info["classes"] == CLASS_NAMES and info["model"]["architecture"] == "custom_cnn_sa"
    assert info["metrics"] is None          # no evaluation files in the test setup -> never invented


def test_predict_contract(client):
    r = post(client, png_bytes())
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["prediction"] in CLASS_NAMES
    p = list(b["probabilities"].values())
    assert set(b["probabilities"]) == set(CLASS_NAMES) and all(math.isfinite(v) for v in p)
    assert abs(sum(p) - 1) < 1e-4
    assert b["prediction"] == max(b["probabilities"], key=b["probabilities"].get)
    assert abs(b["confidence"] - max(p)) < 1e-9
    assert b["explanation"]["available"] and b["explanation"]["overlay"].startswith("data:image/png")
    assert len(b["preprocessing_steps"]) >= 2 and b["report"]


def test_inference_is_deterministic(client):
    a, b = (post(client, png_bytes()).json() for _ in range(2))
    assert np.allclose(list(a["probabilities"].values()), list(b["probabilities"].values()), atol=1e-6)


def test_invalid_uploads(client):
    assert post(client, b"", "x.png").status_code == 400
    assert post(client, b"not an image", "x.png").status_code == 422
    assert post(client, png_bytes(), "x.gif", "image/gif").status_code == 415
    assert post(client, b"%PDF-1.4 fake", "x.png").status_code in (415, 422)
    assert client.post("/api/predict").status_code == 400
    r = post(client, png_bytes(size=40))                       # below minimum size
    assert r.status_code == 422 and "too small" in r.json()["error"]


def test_blank_image_rejected(client):
    import io
    from PIL import Image
    buf = io.BytesIO(); Image.new("L", (300, 300), 128).save(buf, "PNG")
    assert post(client, buf.getvalue()).status_code == 422


def test_quality_warning_low_resolution(client):
    b = post(client, png_bytes(size=100)).json()
    assert any(w["code"] == "low_resolution" for w in b["quality_warnings"])


def test_report_pdf_contains_real_prediction(client):
    b = post(client, png_bytes(), "my scan.png").json()
    r = client.post("/api/report", json={"prediction_id": b["prediction_id"]})
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content[:5] == b"%PDF-"
    from io import BytesIO
    from pypdf import PdfReader
    text = " ".join(pg.extract_text() for pg in PdfReader(BytesIO(r.content)).pages)
    assert b["prediction_id"] in text and b["prediction"] in text and "my scan.png" in text
    assert f"{b['confidence']:.1%}" in text and "not a diagnosis" in text.lower()
    assert client.post("/api/report", json={"prediction_id": "unknown"}).status_code == 404


def test_missing_model_is_reported_clearly(empty_client):
    h = empty_client.get("/api/health").json()
    assert h["status"] == "healthy" and h["model_loaded"] is False and "Missing" in h["detail"]
    r = post(empty_client, png_bytes())
    assert r.status_code == 503 and "not available" in r.json()["error"]
    assert "Traceback" not in r.text and "/tmp" not in r.text
    info = empty_client.get("/api/model-info").json()
    assert info["model_loaded"] is False and info["metrics"] is None


def test_corrupt_weights_do_not_crash_startup(tiny_model_dir, tmp_path):
    import shutil
    from tests.conftest import _client
    d = tmp_path / "m"; shutil.copytree(tiny_model_dir, d)
    (d / "final_model.h5").write_bytes(b"garbage")
    with _client(d, tmp_path / "r") as c:
        assert c.get("/api/health").json()["model_loaded"] is False
        assert post(c, png_bytes()).status_code == 503
