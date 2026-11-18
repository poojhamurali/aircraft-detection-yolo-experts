"""Train and evaluate a YOLOv8 detector on the prepared IRdata-4465 splits.

Hyperparameters are adjusted for single-band infrared imagery with small targets:
colour jitter is disabled (the frames are grayscale), brightness jitter and
horizontal flips are kept, and scale/rotation augmentation is reduced so that
targets a few dozen pixels wide are not destroyed.

Usage:
    python scripts/train_eval.py --data datasets/seq_disjoint/data.yaml \
        --model yolov8n.pt --epochs 10 --name ir_yolov8n_clean
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["airplane", "bird", "drone", "helicopter"]

# Infrared-specific augmentation recipe.
IR_HYP = dict(
    hsv_h=0.0,        # no hue information in a single infrared band
    hsv_s=0.0,        # no saturation information either
    hsv_v=0.3,        # thermal gain/level does vary between sequences
    degrees=0.0,      # sky scenes have a consistent horizon
    translate=0.10,
    scale=0.30,       # mild: aggressive zoom-out erases few-pixel targets
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    close_mosaic=3,   # last 3 epochs on un-tiled images
    mixup=0.0,
    copy_paste=0.0,
)


def require_gpu(device: str, allow_cpu: bool):
    """Refuse to train on CPU by accident.

    Metal is invisible inside a sandbox, where ultralytics silently falls back to
    CPU and a run that should take minutes takes hours.
    """
    if allow_cpu or device == "cpu":
        return
    import torch
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit(
            "MPS requested but unavailable (built={}). Run outside the sandbox, "
            "or pass --allow-cpu to accept CPU training."
            .format(torch.backends.mps.is_built()))
    if device.isdigit() and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but no GPU is visible.")


def metrics_dict(res):
    """Flatten an ultralytics DetMetrics object into plain JSON."""
    out = {
        "mAP50": round(float(res.box.map50), 4),
        "mAP50_95": round(float(res.box.map), 4),
        "precision": round(float(res.box.mp), 4),
        "recall": round(float(res.box.mr), 4),
        "per_class": {},
    }
    for i, c in enumerate(res.box.ap_class_index):
        p, r, ap50, ap = res.box.class_result(i)
        out["per_class"][CLASS_NAMES[int(c)]] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "mAP50": round(float(ap50), 4),
            "mAP50_95": round(float(ap), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--weights", default=None,
                    help="COCO weights to transfer into a .yaml architecture (e.g. P2 variants)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "False"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--predict-samples", type=int, default=9)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit CPU training; without it a missing GPU is a hard error")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing run directory instead of refusing")
    args = ap.parse_args()

    from ultralytics import YOLO

    require_gpu(args.device, args.allow_cpu)

    run_dir = HERE / "outputs" / "runs" / args.name
    if (run_dir / "results.csv").exists() and not args.force:
        raise SystemExit(
            f"{run_dir} already holds a finished run. Two trainings writing to one "
            f"directory interleave their results.csv and corrupt both. Delete it or "
            f"pass --force.")

    data = str(Path(args.data).resolve())
    project = str(HERE / "outputs" / "runs")

    model = YOLO(args.model)
    if args.weights:
        model = model.load(args.weights)
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        cache=False if args.cache == "False" else args.cache,
        workers=args.workers,
        project=project,
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        deterministic=True,
        plots=True,
        val=True,
        **IR_HYP,
    )

    best = run_dir / "weights" / "best.pt"
    trained = YOLO(str(best))

    results = {"run": args.name, "data": data, "model": args.model,
               "epochs": args.epochs, "imgsz": args.imgsz, "seed": args.seed}
    for split in ("val", "test"):
        res = trained.val(data=data, split=split, imgsz=args.imgsz, batch=args.batch,
                          device=args.device, project=project,
                          name=f"{args.name}_{split}_eval", exist_ok=True, plots=True)
        results[split] = metrics_dict(res)

    # qualitative check on held-out test frames, spread across the split
    test_images = sorted((Path(dataset_root(data)) / "images" / "test").glob("*.jpg"))
    if test_images:
        step = max(1, len(test_images) // args.predict_samples)
        samples = test_images[::step][:args.predict_samples]
        trained.predict(source=[str(p) for p in samples],
                        imgsz=args.imgsz, device=args.device, conf=0.25, save=True,
                        project=project, name=f"{args.name}_predictions", exist_ok=True)

    out = run_dir / "metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nmetrics -> {out}")
    print(json.dumps({k: results[k] for k in ("val", "test")}, indent=2))


def dataset_root(data_yaml: str) -> str:
    """Dataset root recorded in the generated data.yaml."""
    for line in Path(data_yaml).read_text().splitlines():
        if line.startswith("path:"):
            return line.split(":", 1)[1].strip()
    return str(Path(data_yaml).parent)


if __name__ == "__main__":
    main()
