#!/usr/bin/env bash
# Quantify what airport overlap is worth, then report the honest baseline.
#
# Two runs, identical in model, epoch budget, seed and augmentation. The only
# difference is whether an airport may appear in both training and test:
#
#   random_split      airports mixed across splits (what ../dataset shipped)
#   airport_disjoint  no airport in more than one split (what the proposal promises)
#
# The gap between them is the cost of airport overlap, measured rather than
# asserted. Only the airport_disjoint number should be quoted as a result.
#
# Must run outside a sandbox: Metal is invisible to a sandboxed process and
# ultralytics then falls back to CPU silently.
#
# Usage: bash scripts/run_splits.sh [epochs] [model]

set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${1:-25}"
MODEL="${2:-yolov8s.pt}"
# Batch is stated here rather than left to the default so both runs provably
# share it. Batch size changes the gradient noise and therefore the final mAP,
# so a comparison across two different batch sizes measures nothing.
BATCH="${3:-4}"
PY=../.venv/bin/python

# The lock records its owner's pid and is reclaimed automatically if that pid is
# gone. An earlier version held no pid, so telling a live lock from a stale one
# meant deleting it by hand -- and deleting it while its owner was still running
# let a second trainer into the same run directory, where the two interleaved
# their results.csv and destroyed each other's weights.
mkdir -p outputs
LOCK=outputs/.splits.lock
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

run () {  # name  data.yaml
    local name="$1" data="$2"
    local dir="outputs/runs/$name"
    echo "=== $name  ($MODEL, ${EPOCHS} epochs, batch ${BATCH}) ==="
    if [ -f "$dir/metrics.json" ]; then
        echo "already finished; skipping"
        return
    fi
    # An unfinished directory is removed rather than trained over with --force:
    # train_eval.py refuses to reuse a directory holding a results.csv precisely
    # because two trainings sharing one interleave their rows and corrupt both.
    rm -rf "$dir" "${dir}_val_eval" "${dir}_test_eval" "${dir}_predictions"
    $PY scripts/train_eval.py --data "$data" --model "$MODEL" \
        --epochs "$EPOCHS" --batch "$BATCH" --name "$name" --device mps
}

run random_split_yolov8s      datasets/random_split/data.yaml
run airport_disjoint_yolov8s  datasets/airport_disjoint/data.yaml
