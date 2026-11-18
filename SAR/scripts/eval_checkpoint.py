"""Re-score an existing run's best.pt on val and test without retraining.

Used when training ended outside train_eval.py (interrupted, resumed, or killed),
so metrics.json would otherwise describe an older checkpoint. The previous
metrics.json is kept as metrics_epoch<N>.json, N = the epochs it covered.

Usage (outside the sandbox, so Metal is visible):
    ../.venv/bin/python scripts/eval_checkpoint.py --name clean_yolov8s
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    from ultralytics import YOLO

    run_dir = HERE / "outputs" / "runs" / args.name
    prev = json.loads((run_dir / "metrics.json").read_text())
    with (run_dir / "results.csv").open() as fh:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]

    def fitness(r):
        return 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])

    best = max(rows, key=fitness)
    data, imgsz = prev["data"], prev.get("imgsz", 512)
    planned = prev.get("epochs_planned", prev.get("epochs"))

    model = YOLO(str(run_dir / "weights" / "best.pt"))
    out = {"run": args.name, "data": data, "model": prev.get("model"),
           "epochs_planned": planned, "epochs_completed": len(rows),
           "best_epoch": int(float(best["epoch"])), "imgsz": imgsz, "batch": args.batch}
    if len(rows) < (planned or 0):
        out["note"] = (f"Training stopped after epoch {len(rows)} of {planned}; metrics are "
                       f"from the best checkpoint among those epochs.")
    for split in ("val", "test"):
        res = model.val(data=data, split=split, imgsz=imgsz, batch=args.batch,
                        device=args.device, project=str(run_dir.parent),
                        name=f"{args.name}_{split}_eval", exist_ok=True, plots=True)
        out[split] = {
            "mAP50": round(float(res.box.map50), 4),
            "mAP50_95": round(float(res.box.map), 4),
            "precision": round(float(res.box.mp), 4),
            "recall": round(float(res.box.mr), 4),
        }

    prev_tag = prev.get("epochs_completed", prev.get("epochs", "prev"))
    shutil.copyfile(run_dir / "metrics.json", run_dir / f"metrics_epoch{prev_tag}.json")
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
