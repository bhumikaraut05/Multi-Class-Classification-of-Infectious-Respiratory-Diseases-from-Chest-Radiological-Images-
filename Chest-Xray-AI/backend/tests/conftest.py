import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from tests.synthetic import png_bytes  # noqa: E402


@pytest.fixture()
def png():
    return png_bytes()


def _client(model_dir, reports_dir):
    import importlib
    os.environ["MODEL_DIR"], os.environ["REPORTS_DIR"] = str(model_dir), str(reports_dir)
    os.environ["FRONTEND_DIST"] = str(model_dir / "no-dist")
    from fastapi.testclient import TestClient
    import app.main as main
    importlib.reload(main)
    return TestClient(main.app)


def make_model_dir(d, arch="custom_cnn_sa"):
    """TEST-ONLY artifacts: a randomly initialised network exported exactly like ml.finalize does.
    Its predictions are meaningless; it exists to exercise load -> preprocess -> infer -> explain -> report."""
    from app.config import CLASS_NAMES, CONFIG_FILENAME, MODEL_FILENAME
    from app.preprocessing import PRESETS
    from ml.models import EXPLAIN, build_model
    m = build_model(arch, 224)
    m.save_weights(str(d / MODEL_FILENAME))
    method, layer = EXPLAIN[arch]
    (d / CONFIG_FILENAME).write_text(json.dumps({
        "model_version": "test", "architecture": arch, "display_name": "TEST random-init " + arch,
        "class_names": CLASS_NAMES, "input_size": 224, "preprocessing": PRESETS["clahe"], "mean_image": None,
        "explainability": {"method": method, "layer": layer}, "weights_file": MODEL_FILENAME,
        "parameters": int(m.count_params())}))
    return d


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    return make_model_dir(tmp_path_factory.mktemp("model"))


@pytest.fixture(scope="session")
def client(tiny_model_dir, tmp_path_factory):
    with _client(tiny_model_dir, tmp_path_factory.mktemp("rep")) as c:
        yield c


@pytest.fixture()
def empty_client(tmp_path):
    with _client(tmp_path / "nothing", tmp_path / "rep") as c:
        yield c
