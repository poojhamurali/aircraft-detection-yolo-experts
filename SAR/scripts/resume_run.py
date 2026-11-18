"""Finish an interrupted training run from its last.pt, with the same settings.

Ultralytics stores the optimizer, EMA and epoch counter in last.pt until a run
completes, so resuming continues the planned schedule rather than starting a new
one. The run directory and its results.csv are reused on purpose: this is the
same training, not a second one.

Afterwards re-score the checkpoint with eval_checkpoint.py, since train_eval.py's
final evaluation never ran for an interrupted run.

Usage (outside the sandbox, so Metal is visible):
    ../.venv/bin/python scripts/resume_run.py --name clean_yolov8s
    ../.venv/bin/python scripts/eval_checkpoint.py --name clean_yolov8s
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    import mps_tal_patch
    from train_eval import require_gpu

    last = HERE / "outputs" / "runs" / args.name / "weights" / "last.pt"
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    if ckpt.get("optimizer") is None:
        raise SystemExit(f"{last} has no optimizer state: the run already finished.")
    planned = ckpt["train_args"]["epochs"]
    print(f"resuming {args.name}: epoch {ckpt['epoch'] + 1} of {planned} done")

    if mps_tal_patch.apply(args.device):
        print("applied MPS task-aligned-assigner workaround")
    require_gpu(args.device, allow_cpu=False)

    YOLO(str(last)).train(resume=True)


if __name__ == "__main__":
    main()
