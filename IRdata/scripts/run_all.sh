#!/usr/bin/env bash
# Three 10-epoch experiments on IRdata-4465.
#   A  as-shipped split, YOLOv8n      -> shows the inflation caused by duplicate frames
#   B  sequence-disjoint split, YOLOv8n
#   C  sequence-disjoint split, YOLOv8s
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
EPOCHS=10

echo "===== A: official (as-shipped) split, yolov8n ====="
$PY scripts/train_eval.py --data datasets/official/data.yaml \
    --model yolov8n.pt --epochs $EPOCHS --name A_official_yolov8n

echo "===== B: sequence-disjoint split, yolov8n ====="
$PY scripts/train_eval.py --data datasets/seq_disjoint/data.yaml \
    --model yolov8n.pt --epochs $EPOCHS --name B_clean_yolov8n

echo "===== C: sequence-disjoint split, yolov8s ====="
$PY scripts/train_eval.py --data datasets/seq_disjoint/data.yaml \
    --model yolov8s.pt --epochs $EPOCHS --name C_clean_yolov8s

echo "===== all runs complete ====="
