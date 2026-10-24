"""Audit the dataset BEFORE training: counts, balance, sizes, corrupt files, duplicates, leakage.

    python -m ml.prepare_dataset --data /path/to/dataset
Writes reports/dataset_report.json.  (Patient IDs are not available, so patient-level leakage cannot be checked.)
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter

import numpy as np

from .common import REPORTS_DIR, write_json
from .data import class_counts, load_splits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-hash", type=int, default=None, help="limit images hashed per split (speed)")
    a = ap.parse_args()
    splits = load_splits(a.data, a.seed)
    report = {"data": str(a.data), "splits": {}, "warnings": []}
    seen: dict[str, str] = {}
    cross = 0
    for name, src in splits.items():
        n = len(src)
        shapes, corrupt, dup_in = Counter(), [], 0
        local: set[str] = set()
        for i in range(n if not a.max_hash else min(n, a.max_hash)):
            try:
                img = src.get(i)
            except Exception:  # noqa: BLE001
                corrupt.append(src.names[i]); continue
            shapes[f"{img.shape[1]}x{img.shape[0]}"] += 1
            h = hashlib.sha1(img.tobytes()).hexdigest()
            dup_in += h in local
            if h in seen and seen[h] != name:
                cross += 1
            local.add(h); seen.setdefault(h, name)
        counts = class_counts(src.labels)
        report["splits"][name] = {"n": n, "class_counts": counts, "image_sizes": dict(shapes.most_common(5)),
                                  "corrupt_files": corrupt, "duplicates_within_split": dup_in}
        nz = [c for c in counts.values() if c]
        if nz and max(nz) / min(nz) > 1.5:
            report["warnings"].append(f"{name}: class imbalance {max(nz) / min(nz):.1f}x -> class weights will be used")
        if len(nz) < 5:
            report["warnings"].append(f"{name}: only {len(nz)} of 5 classes present")
    report["duplicates_across_splits"] = cross
    if cross:
        report["warnings"].append(f"{cross} images appear in more than one split (leakage!). Remove them before training.")
    report["warnings"].append("No patient identifiers available: patient-level leakage cannot be excluded.")
    write_json(REPORTS_DIR / "dataset_report.json", report)
    for k, v in report["splits"].items():
        print(k, v["n"], v["class_counts"])
    print("\n".join("WARNING: " + w for w in report["warnings"]))


if __name__ == "__main__":
    main()
