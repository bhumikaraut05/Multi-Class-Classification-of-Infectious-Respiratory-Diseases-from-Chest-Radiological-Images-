"""Dataset access: official splits (HDF5 or class folders), auditing, tf.data pipelines.

Supported layouts for --data:
  A) HDF5 (the original project):  <dir>/Dataset5_raw_train.h5, _val.h5, _test.h5
     with datasets  image (N,224,224,3 uint8), image_label (N,) or one-hot, image_name.
  B) Folders with official splits:  <dir>/{train,val,test}/<ClassName>/*.png|jpg
  C) Folders without splits:        <dir>/<ClassName>/*.png|jpg  -> stratified 70/15/15 split,
     identical files (same SHA-1) are kept together; saved to reports/splits.json.
Patient identifiers are not part of the dataset, so patient-level separation cannot be enforced
(documented limitation).
"""
from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path

import cv2
import numpy as np

from app.config import CLASS_NAMES
from .common import REPORTS_DIR, read_json, write_json

IMG_EXT = {".png", ".jpg", ".jpeg"}
_ALIASES = {
    "covid19": 0, "covid": 0,
    "lungopacity": 1, "opacity": 1,
    "normal": 2,
    "viralpneumonia": 3, "pneumonia": 3, "viral": 3,
    "tuberculosis": 4, "tb": 4,
}


def class_from_folder(name: str) -> int | None:
    return _ALIASES.get(re.sub(r"[^a-z0-9]", "", name.lower()))


class ImageSource:
    """Random access to uint8 RGB images + integer labels."""
    labels: np.ndarray
    names: list[str]

    def __len__(self) -> int:
        return len(self.labels)

    def get(self, i: int) -> np.ndarray:
        raise NotImplementedError


class H5Source(ImageSource):
    def __init__(self, path: Path):
        import h5py
        self.path = Path(path)
        self._f = h5py.File(self.path, "r")
        self._lock = threading.Lock()
        lab = np.asarray(self._f["image_label"])
        self.labels = (lab.argmax(1) if lab.ndim > 1 else lab).astype(int)
        self.names = [n.decode() if isinstance(n, bytes) else str(n) for n in np.asarray(self._f["image_name"])] \
            if "image_name" in self._f else [f"{self.path.stem}_{i}" for i in range(len(self.labels))]

    def get(self, i: int) -> np.ndarray:
        with self._lock:
            img = np.asarray(self._f["image"][i])
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        return img.astype(np.uint8)


class FileSource(ImageSource):
    def __init__(self, files: list[Path], labels: list[int], root: Path | None = None):
        self.files = [Path(f) for f in files]
        self.labels = np.asarray(labels, dtype=int)
        self.names = [str(f.relative_to(root)) if root else f.name for f in self.files]

    def get(self, i: int) -> np.ndarray:
        arr = cv2.imread(str(self.files[i]), cv2.IMREAD_COLOR)
        if arr is None:
            raise IOError(f"unreadable image: {self.files[i]}")
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _scan_class_dir(root: Path) -> tuple[list[Path], list[int]]:
    files, labels = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        c = class_from_folder(d.name)
        if c is None:
            continue
        for f in sorted(d.rglob("*")):
            if f.suffix.lower() in IMG_EXT:
                files.append(f)
                labels.append(c)
    return files, labels


def _file_hash(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def load_splits(data: str | Path, seed: int = 42) -> dict[str, ImageSource]:
    """Return {'train','val','test'} -> ImageSource, respecting official splits when present."""
    root = Path(data)
    if not root.exists():
        raise SystemExit(f"Dataset path not found: {root}\nSee README.md > 'Dataset' for how to obtain it.")
    h5 = {s: next(iter(root.glob(f"*{s}*.h5")), None) for s in ("train", "val", "test")} if root.is_dir() else {}
    if all(h5.values()):
        return {s: H5Source(p) for s, p in h5.items()}
    if root.is_dir() and all((root / s).is_dir() for s in ("train", "val", "test")):
        out = {}
        for s in ("train", "val", "test"):
            f, l = _scan_class_dir(root / s)
            out[s] = FileSource(f, l, root)
        return out
    if root.is_dir():
        files, labels = _scan_class_dir(root)
        if not files:
            raise SystemExit(f"No class folders with images found under {root}. Expected folders such as "
                             "COVID-19/, Lung-Opacity/, Normal/, Viral Pneumonia/, Tuberculosis/.")
        return _random_split(root, files, labels, seed)
    raise SystemExit("Unsupported --data path: expected a directory.")


def _random_split(root: Path, files: list[Path], labels: list[int], seed: int) -> dict[str, ImageSource]:
    from sklearn.model_selection import StratifiedGroupKFold
    saved = REPORTS_DIR / "splits.json"
    rel = [str(f.relative_to(root)) for f in files]
    if saved.exists():
        sp = read_json(saved)
        if sp.get("root_files") == len(files) and sp.get("seed") == seed:
            idx = {r: i for i, r in enumerate(rel)}
            return {s: FileSource([files[idx[r]] for r in sp[s]], [labels[idx[r]] for r in sp[s]], root)
                    for s in ("train", "val", "test")}
    hashes = [_file_hash(f) for f in files]
    groups = {h: i for i, h in enumerate(dict.fromkeys(hashes))}
    g = np.array([groups[h] for h in hashes])
    y = np.array(labels)
    # 70/15/15 via two grouped, stratified splits (identical files never straddle splits).
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    folds = [te for _, te in sgkf.split(np.zeros(len(y)), y, g)]
    test_idx = np.concatenate(folds[:3])
    val_idx = np.concatenate(folds[3:6])
    train_idx = np.concatenate(folds[6:])
    sp = {"seed": seed, "root_files": len(files)}
    out = {}
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        idx = np.sort(idx)
        sp[name] = [rel[i] for i in idx]
        out[name] = FileSource([files[i] for i in idx], [labels[i] for i in idx], root)
    write_json(saved, sp)
    return out


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(len(CLASS_NAMES))}


def class_weights_if_needed(labels: np.ndarray, ratio_threshold: float) -> tuple[dict[int, float] | None, dict]:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(float)
    nz = counts[counts > 0]
    ratio = float(nz.max() / nz.min())
    info = {"counts": counts.tolist(), "imbalance_ratio": ratio, "used": ratio > ratio_threshold}
    if ratio <= ratio_threshold:
        return None, info
    w = {i: float(counts.sum() / (len(nz) * c)) if c > 0 else 1.0 for i, c in enumerate(counts)}
    info["weights"] = w
    return w, info


# ----------------------------------------------------------------------- augmentation / tf.data
def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild geometric + intensity augmentation for TRAINING images only.  No flips (laterality matters)."""
    h, w = img.shape[:2]
    ang = rng.uniform(-10, 10)
    scale = rng.uniform(0.92, 1.08)
    tx, ty = rng.uniform(-0.05, 0.05, 2) * (w, h)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    m[:, 2] += (tx, ty)
    out = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    alpha, beta = rng.uniform(0.9, 1.1), rng.uniform(-10, 10)
    return np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def make_dataset(source: ImageSource, indices: np.ndarray, cfg: dict, mean: np.ndarray | None,
                 batch: int, training: bool, seed: int = 42):
    import tensorflow as tf
    from app.preprocessing import preprocess_array
    size = int(cfg["input_size"])
    n_cls = len(CLASS_NAMES)
    rng = np.random.default_rng(seed)

    def load(i):
        i = int(i)
        img = source.get(i)
        if training:
            img = augment(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), rng)
        x, _ = preprocess_array(img, cfg, mean)
        y = np.zeros(n_cls, np.float32)
        y[source.labels[i]] = 1.0
        return x.astype(np.float32), y

    def fn(i):
        x, y = tf.numpy_function(load, [i], [tf.float32, tf.float32])
        x.set_shape((size, size, 3))
        y.set_shape((n_cls,))
        return x, y

    ds = tf.data.Dataset.from_tensor_slices(np.asarray(indices, dtype=np.int64))
    if training:
        ds = ds.shuffle(len(indices), seed=seed, reshuffle_each_iteration=True)
    return ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def subsample(source: ImageSource, limit: int | None, seed: int = 0) -> np.ndarray:
    """Indices of the split, optionally a stratified-ish random subset (for smoke tests only)."""
    idx = np.arange(len(source))
    if limit and limit < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(idx, limit, replace=False))
    return idx
