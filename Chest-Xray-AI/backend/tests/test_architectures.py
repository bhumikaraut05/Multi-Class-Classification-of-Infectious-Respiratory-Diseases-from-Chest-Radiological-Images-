"""Every candidate architecture must round-trip (save -> load in the API) and produce an explanation."""
import pytest

from tests.conftest import _client, make_model_dir
from tests.synthetic import png_bytes


@pytest.mark.parametrize("arch", ["baseline_cnn", "vgg19", "resnet50", "vit"])
def test_architecture_serves_prediction_and_explanation(arch, tmp_path):
    d = make_model_dir(tmp_path / "m", arch) if (tmp_path / "m").mkdir() is None else None
    with _client(d, tmp_path / "r") as c:
        assert c.get("/api/health").json()["model_loaded"] is True
        r = c.post("/api/predict", files={"file": ("x.png", png_bytes(), "image/png")})
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["explanation"]["available"], arch
        assert abs(sum(b["probabilities"].values()) - 1) < 1e-4
        if arch == "vit":
            assert c.get("/api/model-info").json()["model"]["parameters"] == 89_710_422
