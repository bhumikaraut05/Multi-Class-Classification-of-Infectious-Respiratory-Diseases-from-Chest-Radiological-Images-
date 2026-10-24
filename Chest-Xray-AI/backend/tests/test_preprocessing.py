import numpy as np

from app.preprocessing import PRESETS, preprocess_array


def test_preprocessing_is_deterministic_and_shaped():
    img = (np.random.RandomState(0).rand(300, 280, 3) * 255).astype(np.uint8)
    for name, cfg in PRESETS.items():
        mean = np.full((224, 224, 3), 0.4, np.float32) if cfg["mean_subtract"] else None
        a, _ = preprocess_array(img, cfg, mean)
        b, _ = preprocess_array(img, cfg, mean)
        assert a.shape == (224, 224, 3) and a.dtype == np.float32 and np.array_equal(a, b), name


def test_training_and_inference_share_the_same_function():
    import inspect
    import ml.data
    assert "preprocess_array" in inspect.getsource(ml.data.make_dataset)
