#!/usr/bin/env bash
# IR improvement study: YOLO11s, 15 epochs, leakage-free `clean` split, Apple GPU.
#
#   base640   reference recipe at 640 px                      seeds 0 1 2
#   img960    trained and evaluated at 960 px                 seeds 0 1 2
#   img1280   trained and evaluated at 1280 px                seed 0 (4x the cost)
#   hard640   airplane/bird + mined hard frames oversampled   seeds 0 1 2
#
# Each run is followed by cached low-threshold inference (improve_eval.py predict);
# base640 additionally gets tiled inference and a 960 px test-time pass, which need
# no retraining. Sequence post-processing and the confidence sweep run afterwards
# from the caches (summarize_improvements.py).
#
# Must run outside a sandbox (Metal is invisible inside one). Runs are idempotent:
# a finished run is skipped, an unfinished one is deleted and retrained.
#
# Usage: bash scripts/run_improvements.sh [epochs]

set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${1:-15}"
PY=../.venv/bin/python
export YOLO_CONFIG_DIR="$PWD/../outputs/.cache" MPLCONFIGDIR="$PWD/../outputs/.cache"

LOCK=outputs/.improvements.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "a run is already in progress (lock: $LOCK). Remove it if that is stale." >&2
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

train () {  # name data imgsz batch seed
    local name="$1" data="$2" imgsz="$3" batch="$4" seed="$5"
    local dir="outputs/runs/$name"
    echo "=== $name  (yolo11s, $data, ${imgsz}px, batch $batch, seed $seed, $EPOCHS ep) $(date +%H:%M) ==="
    if [ -f "$dir/metrics.json" ]; then
        echo "already finished; skipping training"
    else
        rm -rf "$dir" "${dir}_val_eval" "${dir}_test_eval" "${dir}_predictions"
        $PY scripts/train_eval.py --data "$data" --model yolo11s.pt --epochs "$EPOCHS" \
            --imgsz "$imgsz" --batch "$batch" --seed "$seed" --name "$name" --device mps
    fi
    $PY scripts/improve_eval.py predict --run "$name" --imgsz "$imgsz" --mode plain
}

CLEAN=datasets/clean/data.yaml
HARD=datasets/clean_hard/data.yaml

for s in 0 1 2; do
    train "imp_base640_s$s" "$CLEAN" 640 16 "$s"
    $PY scripts/improve_eval.py predict --run "imp_base640_s$s" --imgsz 640 --mode tiled
    $PY scripts/improve_eval.py predict --run "imp_base640_s$s" --imgsz 960 --mode plain
done

[ -f "$HARD" ] || $PY scripts/build_hard_split.py --mine-run imp_base640_s0

train imp_hard640_s0  "$HARD"  640  16 0
train imp_img960_s0   "$CLEAN" 960  16 0
train imp_img1280_s0  "$CLEAN" 1280 8  0
for s in 1 2; do
    train "imp_hard640_s$s" "$HARD"  640 16 "$s"
    train "imp_img960_s$s"  "$CLEAN" 960 16 "$s"
done

$PY scripts/summarize_improvements.py
echo "=== done $(date +%H:%M) ==="
