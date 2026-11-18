#!/usr/bin/env bash
# YOLOv8s reference for the IR improvement study: same recipe as imp_base640
# (640 px, 15 epochs, clean split, seeds 0 1 2), so the project's base architecture
# is measured under the same protocol as YOLO11s.
#
# Waits for run_improvements.sh to release its lock, then takes the lock itself, so
# the two never share the GPU. Re-runs the summary (and weight export) at the end.
#
# Must run outside a sandbox (Metal is invisible inside one).
#
# Usage: bash scripts/run_improvements_v8.sh [epochs]

set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${1:-15}"
PY=../.venv/bin/python
export YOLO_CONFIG_DIR="$PWD/../outputs/.cache" MPLCONFIGDIR="$PWD/../outputs/.cache"

LOCK=outputs/.improvements.lock
until mkdir "$LOCK" 2>/dev/null; do sleep 60; done
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
echo "=== lock acquired $(date +%H:%M) ==="

for s in 0 1 2; do
    name="imp_v8s640_s$s"
    dir="outputs/runs/$name"
    echo "=== $name  (yolov8s, 640px, seed $s, $EPOCHS ep) $(date +%H:%M) ==="
    if [ ! -f "$dir/metrics.json" ]; then
        rm -rf "$dir" "${dir}_val_eval" "${dir}_test_eval" "${dir}_predictions"
        $PY scripts/train_eval.py --data datasets/clean/data.yaml --model yolov8s.pt \
            --epochs "$EPOCHS" --imgsz 640 --batch 16 --seed "$s" --name "$name" --device mps
    fi
    $PY scripts/improve_eval.py predict --run "$name" --imgsz 640 --mode plain
done

$PY scripts/summarize_improvements.py
echo "=== done $(date +%H:%M) ==="
