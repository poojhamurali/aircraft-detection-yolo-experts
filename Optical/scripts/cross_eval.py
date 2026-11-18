"""Score both split models on one common test set.

The two training runs each report a test mAP, but those numbers come from two
different sets of images, so the difference between them measures the choice of
test images as much as it measures leakage. This evaluates both checkpoints on
the airport-disjoint test set instead -- four airports (DAL, DEN, JOH, SUV) that
the airport-disjoint model never saw, and of which the random-split model
trained on 68% outright.

Same images, same metric, two models: the gap is what memorising the test
airports is worth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent.parent
COMMON = HERE / "datasets" / "airport_disjoint" / "data.yaml"

MODELS = {
    "random_split_yolov8s": "trained on 68% of these very images",
    "airport_disjoint_yolov8s": "never saw these airports",
}


def main():
    import mps_tal_patch
    mps_tal_patch.apply("mps")
    from ultralytics import YOLO

    out = {}
    for name, note in MODELS.items():
        weights = HERE / "outputs" / "runs" / name / "weights" / "best.pt"
        if not weights.exists():
            raise SystemExit(f"missing {weights}")
        res = YOLO(str(weights)).val(
            data=str(COMMON), split="test", imgsz=640, batch=4, device="mps",
            project=str(HERE / "outputs" / "runs"),
            name=f"crosseval_{name}", exist_ok=True, plots=False,
        )
        out[name] = {
            "note": note,
            "mAP50": round(float(res.box.map50), 4),
            "mAP50_95": round(float(res.box.map), 4),
            "precision": round(float(res.box.mp), 4),
            "recall": round(float(res.box.mr), 4),
        }

    dest = HERE / "outputs" / "cross_eval.json"
    dest.write_text(json.dumps(out, indent=2))

    print("\n=== both models on the airport-disjoint test set (4 unseen airports) ===")
    for name, m in out.items():
        print(f"{name:<26} mAP50 {m['mAP50']:.4f}  mAP50-95 {m['mAP50_95']:.4f}  "
              f"P {m['precision']:.4f}  R {m['recall']:.4f}   ({m['note']})")
    a, b = (out[k]["mAP50"] for k in MODELS)
    print(f"\nleakage is worth {a - b:+.4f} mAP50 on identical images")
    print(f"metrics -> {dest}")


if __name__ == "__main__":
    main()
