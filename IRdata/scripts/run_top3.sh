#!/usr/bin/env bash
# Train the three architectures that led the 12-epoch nano screen for 25 epochs
# on the leakage-free `clean` split, at the small (s) scale.
#
# Screen ranking (12 ep, nano, test mAP@0.5):
#   yolo12n 0.606 > yolo11n 0.585 > yolov8n-p2 0.579 > yolov8n 0.571
#     = yolo26n 0.571 >> yolo26n-p2 0.335
# so the three families carried forward are yolo12, yolo11 and yolov8 with a P2
# (stride-4) head. The P2 variant is kept despite ranking third because it is the
# only one of the three that adds a detection level matched to these target
# sizes, and 65% of boxes here are under 32x32 px.
#
# Must run outside a sandbox: Metal is invisible to a sandboxed process and
# ultralytics then falls back to CPU silently. train_eval.py --device mps fails
# loudly rather than doing that.
#
# Usage: bash scripts/run_top3.sh [epochs]

set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${1:-25}"
DATA=datasets/clean/data.yaml
PY=../.venv/bin/python

LOCK=outputs/.top3.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "a run is already in progress (lock: $LOCK). Remove it if that is stale." >&2
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

run () {  # name  model  [pretrained-weights-for-yaml-configs]
    local name="$1" model="$2" weights="${3:-}"
    local dir="outputs/runs/$name"
    echo "=== $name  ($model, ${EPOCHS} epochs) ==="

    # Idempotent: a finished run (metrics.json present) is left alone, so the
    # script can be re-invoked after an interruption without discarding hours of
    # completed training.
    if [ -f "$dir/metrics.json" ]; then
        echo "already finished; skipping training"
    else
        # An unfinished directory is deleted rather than trained over with
        # --force. train_eval.py refuses to write into a directory that already
        # holds a results.csv precisely because two trainings sharing one
        # directory interleave their rows and corrupt both; passing --force here
        # would disable that guard for exactly the case it exists to catch.
        rm -rf "$dir" "${dir}_val_eval" "${dir}_test_eval" "${dir}_predictions"
        if [ -n "$weights" ]; then
            $PY scripts/train_eval.py --data "$DATA" --model "$model" --weights "$weights" \
                --epochs "$EPOCHS" --name "$name" --device mps
        else
            $PY scripts/train_eval.py --data "$DATA" --model "$model" \
                --epochs "$EPOCHS" --name "$name" --device mps
        fi
    fi
    $PY scripts/confusion_analysis.py --run "$name" --split test
}

run top1_yolo12s   yolo12s.pt
run top2_yolo11s   yolo11s.pt
run top3_yolov8s_p2 yolov8s-p2.yaml yolov8s.pt

$PY scripts/collect_results.py
