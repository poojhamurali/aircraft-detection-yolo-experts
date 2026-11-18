#!/usr/bin/env bash
# Train on the SAR aircraft splits, one run at a time.
#
# Two variants, identical in model, epochs, batch, seed and augmentation. Only
# the split differs:
#
#   official  the split as SARDet-100K ships it; 4.6% of its evaluation images
#             have a near-identical twin in its training set
#   clean     no near-duplicate group spans two splits
#
# The gap between them is what that contamination is worth. Only the clean
# number should be quoted as a result.
#
# The lock records its owner's pid and is reclaimed only if that pid is gone.
# Without it two trainings can share a run directory, interleave their
# results.csv rows and destroy each other's weights -- which has happened twice
# in this project, once on the infrared stream and once here.
#
# Must run outside a sandbox: Metal is invisible to a sandboxed process and
# ultralytics then falls back to CPU silently.
#
# Usage: bash scripts/run_splits.sh [epochs] [model] [batch] [variants...]

set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${1:-15}"
MODEL="${2:-yolov8s.pt}"
BATCH="${3:-8}"
shift 3 2>/dev/null || true
VARIANTS="${*:-clean official}"
PY=../.venv/bin/python

mkdir -p outputs
LOCK=outputs/.runs.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    owner=$(cat "$LOCK/pid" 2>/dev/null || echo "")
    if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
        echo "a run is already in progress (pid $owner). Wait for it or kill it." >&2
        exit 1
    fi
    echo "reclaiming stale lock from dead pid ${owner:-unknown}" >&2
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK" 2>/dev/null || true' EXIT

for variant in $VARIANTS; do
    name="${variant}_${MODEL%.pt}"
    dir="outputs/runs/$name"
    echo "=== $name  ($MODEL, ${EPOCHS} epochs, batch ${BATCH}, split ${variant}) ==="

    if [ -f "$dir/metrics.json" ]; then
        echo "already finished; skipping"
        continue
    fi
    # An unfinished directory is removed rather than trained over with --force:
    # train_eval.py refuses to reuse a directory holding a results.csv precisely
    # because two trainings sharing one interleave their rows and corrupt both.
    rm -rf "$dir" "${dir}_val_eval" "${dir}_test_eval" "${dir}_predictions"

    $PY scripts/train_eval.py \
        --data "datasets/${variant}/data.yaml" --model "$MODEL" \
        --epochs "$EPOCHS" --batch "$BATCH" --name "$name" --device mps || {
            echo "run $name failed" >&2; exit 1; }
done

echo
echo "all runs finished:"
for variant in $VARIANTS; do
    name="${variant}_${MODEL%.pt}"
    [ -f "outputs/runs/$name/metrics.json" ] && echo "  $name -> outputs/runs/$name/metrics.json"
done
