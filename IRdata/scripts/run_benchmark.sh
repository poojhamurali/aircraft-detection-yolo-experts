#!/usr/bin/env bash
# Architecture benchmark on the clean sequence-disjoint split.
#
# Every run uses identical data, image size, batch, seed and augmentation, so the
# only variable is the architecture. All runs use the Apple GPU (MPS); training
# aborts rather than silently falling back to CPU.
#
#   stage1  YOLOv8 vs YOLO11 vs YOLO12 vs YOLO26      (architecture only)
#   stage2  the same backbones with a P2 stride-4 head (small-target capacity)
#
# Nano scale is used for screening so the whole comparison fits in ~85 min on an
# M5 Pro; the winner is then retrained at 's' scale for the headline number.
#
# Usage: bash scripts/run_benchmark.sh [stage1|stage2|all] [epochs] [scale]
set -euo pipefail
cd "$(dirname "$0")/.."

# Single-instance lock. Two concurrent benchmarks write to the same run
# directories, interleave their results.csv rows and corrupt both sets of
# metrics, while also halving GPU throughput.
LOCK=outputs/.benchmark.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "a benchmark is already running (lock: $LOCK). Remove it if that is stale." >&2
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

PY=../.venv/bin/python
DATA=datasets/seq_disjoint/data.yaml
STAGE="${1:-all}"
EPOCHS="${2:-12}"
S="${3:-n}"

run() {  # run <name> <model> [pretrained-weights]
    echo "===== ${1} (${EPOCHS} epochs, scale ${S}, MPS) ====="
    $PY scripts/train_eval.py --data "$DATA" --model "$2" \
        ${3:+--weights "$3"} \
        --epochs "$EPOCHS" --device mps --batch 16 --imgsz 640 --cache ram \
        --name "$1"
}

stage1() {
    run "bench_yolov8${S}" "yolov8${S}.pt"
    run "bench_yolo11${S}" "yolo11${S}.pt"
    run "bench_yolo12${S}" "yolo12${S}.pt"
    run "bench_yolo26${S}" "yolo26${S}.pt"
}

stage2() {
    run "bench_yolov8${S}_p2" "yolov8${S}-p2.yaml" "yolov8${S}.pt"
    run "bench_yolo26${S}_p2" "yolo26${S}-p2.yaml" "yolo26${S}.pt"
}

case "$STAGE" in
  stage1) stage1 ;;
  stage2) stage2 ;;
  all)    stage1; stage2 ;;
  *) echo "unknown stage: $STAGE" >&2; exit 1 ;;
esac

echo "===== ${STAGE} complete ====="
