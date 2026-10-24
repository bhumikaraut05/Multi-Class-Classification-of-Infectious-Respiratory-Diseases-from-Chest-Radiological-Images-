"""Train every candidate on the SAME split + preprocessing and pick ONE by VALIDATION macro-F1.

    python -m ml.select_model --data /path/to/dataset
The test set is not touched.  Output: reports/model_selection.{json,md}, reports/selected_model.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import traceback
from pathlib import Path

from .common import CANDIDATES, DISPLAY_NAMES, REPORTS_DIR, read_json, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", default=CANDIDATES, choices=CANDIDATES)
    ap.add_argument("--preprocessing", default="clahe", choices=["plain", "clahe", "legacy"])
    ap.add_argument("--out", default=None, help="parent directory for run folders (default reports/runs)")
    ap.add_argument("--reports", default=str(REPORTS_DIR), help="where to write the selection files")
    ap.add_argument("--epochs", type=int); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="do not initialise VGG19/ResNet50 from ImageNet")
    ap.add_argument("--skip-existing", action="store_true", help="reuse finished runs (resume after interruption)")
    ap.add_argument("--limit", type=int, default=None, help="SMOKE TESTS ONLY")
    a = ap.parse_args()

    out_root = Path(a.out or REPORTS_DIR / "runs")
    rows = []
    for m in a.models:
        run = out_root / f"{m}__{a.preprocessing}"
        try:
            if not (a.skip_existing and (run / "val_metrics.json").exists()):
                # Each candidate runs in its own process, so an out-of-memory kill of one model cannot stop the rest.
                cmd = [sys.executable, "-m", "ml.train", "--data", a.data, "--model", m, "--preprocessing",
                       a.preprocessing, "--out", str(out_root), "--seed", str(a.seed)]
                cmd += ["--epochs", str(a.epochs)] if a.epochs else []
                cmd += ["--no-pretrained"] if a.no_pretrained else []
                cmd += ["--limit", str(a.limit)] if a.limit else []
                code = subprocess.call(cmd)
                if code != 0:
                    raise RuntimeError(f"training process exited with code {code} (negative/137 usually means out of memory)")
            vm, rc = read_json(run / "val_metrics.json"), read_json(run / "run_config.json")
            rows.append({"model": m, "display_name": DISPLAY_NAMES[m], "status": "ok", "run_dir": str(run),
                         "val_accuracy": vm["accuracy"], "val_macro_f1": vm["macro_f1"],
                         "val_weighted_f1": vm["weighted_f1"], "val_macro_auc": vm["macro_auc_ovr"],
                         "min_class_recall": min(v["recall"] for v in vm["per_class"].values()),
                         "parameters": rc["parameters"], "epochs_run": rc["epochs_run"],
                         "train_seconds": rc["train_seconds"]})
        except Exception as exc:  # noqa: BLE001 - a failed candidate must not hide the others
            traceback.print_exc()
            rows.append({"model": m, "display_name": DISPLAY_NAMES[m], "status": f"failed: {type(exc).__name__}: {exc}"})

    ok = sorted([r for r in rows if r["status"] == "ok"], key=lambda r: (r["val_macro_f1"], r["val_accuracy"], r["val_macro_auc"] or 0), reverse=True)
    if not ok:
        raise SystemExit("No candidate trained successfully; see errors above.")
    best = ok[0]
    sel = {"selected_model": best["model"], "display_name": best["display_name"], "preprocessing": a.preprocessing,
           "run_dir": best["run_dir"], "criterion": "highest validation macro-F1 (ties: validation accuracy, then macro-AUC)",
           "selected_on": "validation split only", "created_at": dt.datetime.now().isoformat(timespec="seconds"),
           "smoke_test_only": bool(a.limit)}
    reports = Path(a.reports)
    write_json(reports / "model_selection.json", {"candidates": rows, **sel})
    write_json(reports / "selected_model.json", sel)
    md = ["| Model | Val acc | Val macro-F1 | Val macro-AUC | Min class recall | Params |", "|---|---|---|---|---|---|"]
    md += [f"| {r['display_name']} | {r['val_accuracy']:.4f} | {r['val_macro_f1']:.4f} | {r['val_macro_auc']:.4f} | "
           f"{r['min_class_recall']:.3f} | {r['parameters']:,} |" for r in ok]
    md += [f"| {r['display_name']} | {r['status']} | | | | |" for r in rows if r["status"] != "ok"]
    (reports / "model_selection.md").write_text("\n".join(md) + f"\n\nSelected: **{best['display_name']}**\n")
    print("\n".join(md)); print("\nSELECTED:", best["display_name"])


if __name__ == "__main__":
    main()
