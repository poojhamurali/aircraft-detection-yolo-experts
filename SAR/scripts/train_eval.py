"""Train and evaluate a YOLO detector on the prepared SAR aircraft splits.

Augmentation differs from both earlier streams, for reasons specific to radar.

Colour jitter is off, as in the infrared stream: SAR amplitude is a single
channel replicated to three, so there is no colour to shift.

Rotation is off, which is the opposite of the optical choice. An overhead
optical scene has no canonical "up", so rotating it produces another plausible
scene. A SAR image does have one: range and azimuth are fixed by the sensor
geometry, and shadow, layover and radar bright returns all fall on the side the
sensor dictates. Rotating a SAR patch produces an image that no radar could
have formed, and asks the model to learn a symmetry the physics does not have.
Flips are kept, being the one geometric transform the SAR detection literature
treats as standard.

    ../.venv/bin/python scripts/train_eval.py --data datasets/clean/data.yaml \
        --model yolov8s.pt --epochs 25 --name clean_yolov8s
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["aircraft"]

SAR_HYP = dict(
    hsv_h=0.0,        # SAR amplitude carries no colour
    hsv_s=0.0,
    hsv_v=0.3,        # backscatter level does vary between scenes
    degrees=0.0,      # see module docstring: SAR geometry has a fixed sense
    translate=0.10,
    scale=0.40,
    shear=0.0,
    perspective=0.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    close_mosaic=3,
    mixup=0.0,
    copy_paste=0.0,   # pasting targets ignores speckle statistics
)


def require_gpu(device: str, allow_cpu: bool):
    """Refuse to train on CPU by accident.

    Metal is invisible inside a sandbox, where ultralytics silently falls back
    to CPU and a run that should take an hour takes most of a day.
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
    out = {
        "mAP50": round(float(res.box.map50), 4),
        "mAP50_95": round(float(res.box.map), 4),
        "precision": round(float(res.box.mp), 4),
        "recall": round(float(res.box.mr), 4),
    }
    return out


def dataset_root(data_yaml: str) -> str:
    for line in Path(data_yaml).read_text().splitlines():
        if line.startswith("path:"):
            return line.split(":", 1)[1].strip()
    return str(Path(data_yaml).parent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=25)
    # Most patches are 512x512, so this is their native size: unlike the optical
    # stream nothing is thrown away by resizing.
    ap.add_argument("--imgsz", type=int, default=512)
    # One training patch holds 318 aircraft and mosaic tiles four together, so
    # the assigner's dense intermediates (see mps_tal_patch) grow with batch.
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--cache", default="False", choices=["ram", "disk", "False"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", required=True)
    ap.add_argument("--predict-samples", type=int, default=9)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing run directory instead of refusing")
    args = ap.parse_args()

    from ultralytics import YOLO

    import mps_tal_patch
    if mps_tal_patch.apply(args.device):
        print("applied MPS task-aligned-assigner workaround")

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
    model.train(
        data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, cache=False if args.cache == "False" else args.cache,
        workers=args.workers, project=project, name=args.name, exist_ok=True,
        seed=0, deterministic=True, plots=True, val=True, **SAR_HYP,
    )

    trained = YOLO(str(run_dir / "weights" / "best.pt"))
    results = {"run": args.name, "data": data, "model": args.model,
               "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch}
    for split in ("val", "test"):
        res = trained.val(data=data, split=split, imgsz=args.imgsz, batch=args.batch,
                          device=args.device, project=project,
                          name=f"{args.name}_{split}_eval", exist_ok=True, plots=True)
        results[split] = metrics_dict(res)

    test_images = sorted((Path(dataset_root(data)) / "images" / "test").glob("*.jpg"))
    if test_images and args.predict_samples > 0:
        step = max(1, len(test_images) // args.predict_samples)
        samples = test_images[::step][:args.predict_samples]
        trained.predict(source=[str(p) for p in samples], imgsz=args.imgsz,
                        device=args.device, conf=0.25, save=True, project=project,
                        name=f"{args.name}_predictions", exist_ok=True)

    out = run_dir / "metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nmetrics -> {out}")
    print(json.dumps({k: results[k] for k in ("val", "test")}, indent=2))


if __name__ == "__main__":
    main()
