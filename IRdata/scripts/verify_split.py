"""Independently verify that a materialised YOLO split has no train/eval leakage.

prepare_dataset.py builds the clean split; this script audits the result from the
files on disk instead of trusting that logic, so the guarantee in the report rests
on the artefact that was actually trained on.

Three checks:
  1. no image content (MD5 of pixels) appears in more than one split
  2. no source video sequence contributes frames to more than one split
  3. no near-duplicate pairs across splits, by downscaled pixel difference

Check 3 deliberately avoids a perceptual/average hash. These frames are mostly
uniform sky with a handful of bright pixels on target, so an 8x8 average hash is
near-constant across the whole dataset: 1931 frames collapse into ~534 buckets and
16% of *random* unrelated pairs land within Hamming distance 4. It reports leakage
that is not there. Per-pixel RMSE on a 64x64 thumbnail keeps enough structure to
discriminate -- unrelated frames sit near 61 -- and the random-pair baseline is
reported alongside the result so the threshold can be read in context rather than
taken on faith. The threshold matches prepare_dataset.py's --near-dup-rmse, so
this script verifies the exact property the clean split was built to have.

Usage:
    python scripts/verify_split.py --dataset datasets/seq_disjoint
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
CLASS_NAMES = ["airplane", "bird", "drone", "helicopter"]


def sequence_of(stem: str) -> str:
    """Mirror of prepare_dataset.sequence_of, applied to the materialised stems."""
    import re
    base = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", stem)
    m = re.match(r"^(IR_[A-Z]+_\d+)_frame_\d+$", base)
    if m:
        return m.group(1)
    m = re.match(r"^IR_frame_(\d+)$", base)
    if m:
        return f"IR_UNNAMED_bin{int(m.group(1)) // 100:02d}"
    return f"IR_OTHER_{base}"


def content_hash(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def thumbnail(path: Path):
    """64x64 grayscale thumbnail as float32, for pixel-difference comparison."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "datasets" / "seq_disjoint"))
    ap.add_argument("--rmse-threshold", type=float, default=2.5,
                    help="thumbnail per-pixel RMSE (0-255 scale) below which two frames "
                         "are treated as the same shot; must match "
                         "prepare_dataset.py --near-dup-rmse")
    ap.add_argument("--out", default=str(HERE / "outputs" / "split_verification.json"))
    args = ap.parse_args()

    root = Path(args.dataset).resolve()
    images = {s: sorted((root / "images" / s).glob("*.jpg")) for s in SPLITS}
    if not all(images.values()):
        raise SystemExit(f"{root} does not look like a materialised YOLO split")

    # ---- check 1: exact content overlap ---------------------------------
    by_hash = defaultdict(set)
    hashes = {}
    for split, paths in images.items():
        for p in paths:
            d = content_hash(p)
            hashes[p] = d
            by_hash[d].add(split)
    exact_cross = {d: sorted(s) for d, s in by_hash.items() if len(s) > 1}

    # ---- check 2: sequence overlap --------------------------------------
    seq_splits = defaultdict(set)
    for split, paths in images.items():
        for p in paths:
            seq_splits[sequence_of(p.stem)].add(split)
    seq_cross = {q: sorted(s) for q, s in seq_splits.items() if len(s) > 1}

    # ---- check 3: near-duplicate overlap --------------------------------
    import numpy as np

    def flat(split):
        return np.stack([thumbnail(p).ravel() for p in images[split]]).astype(np.float64)

    train_flat = flat("train")
    train_sq = (train_flat * train_flat).sum(axis=1)
    dim = train_flat.shape[1]

    eval_paths, min_rmse, nearest = [], [], []
    for split in ("val", "test"):
        ev = flat(split)
        ev_sq = (ev * ev).sum(axis=1)
        # errstate: numpy's Accelerate BLAS backend raises spurious divide/overflow
        # warnings from this gemm on macOS; the result is correct.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            gram = ev @ train_flat.T
        d2 = np.maximum(ev_sq[:, None] + train_sq[None, :] - 2.0 * gram, 0.0)
        r = np.sqrt(d2 / dim)
        for i, p in enumerate(images[split]):
            j = int(r[i].argmin())
            eval_paths.append(f"{split}/{p.name}")
            min_rmse.append(float(r[i, j]))
            nearest.append(images["train"][j].name)
    min_rmse = np.array(min_rmse)

    # baseline: how different are two unrelated frames in this dataset at all?
    rng = np.random.default_rng(0)
    idx = rng.choice(len(train_flat), size=(400, 2))
    diff = train_flat[idx[:, 0]] - train_flat[idx[:, 1]]
    baseline = np.sqrt((diff * diff).sum(axis=1) / dim)

    near_eval_files = int((min_rmse < args.rmse_threshold).sum())
    eval_total = len(eval_paths)
    worst = np.argsort(min_rmse)[:15]

    report = {
        "dataset": str(root),
        "images": {s: len(v) for s, v in images.items()},
        "sequences": {s: len({sequence_of(p.stem) for p in v}) for s, v in images.items()},
        "exact_duplicates_across_splits": len(exact_cross),
        "sequences_spanning_splits": len(seq_cross),
        "sequences_spanning_splits_examples": dict(list(seq_cross.items())[:10]),
        "rmse_threshold": args.rmse_threshold,
        "near_duplicate_eval_files": near_eval_files,
        "near_duplicate_eval_pct": round(100.0 * near_eval_files / eval_total, 2),
        "eval_to_train_min_rmse": {
            "median": round(float(np.median(min_rmse)), 2),
            "p05": round(float(np.percentile(min_rmse, 5)), 2),
            "min": round(float(min_rmse.min()), 2),
        },
        "unrelated_train_pair_rmse": {
            "median": round(float(np.median(baseline)), 2),
            "p05": round(float(np.percentile(baseline, 5)), 2),
        },
        "closest_eval_train_pairs": [
            {"eval": eval_paths[i], "train": nearest[i], "rmse": round(float(min_rmse[i]), 2)}
            for i in worst
        ],
    }
    report["leakage_free"] = (report["exact_duplicates_across_splits"] == 0
                              and report["sequences_spanning_splits"] == 0
                              and near_eval_files == 0)

    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"dataset            {root}")
    print(f"images             {report['images']}")
    print(f"sequences          {report['sequences']}")
    print(f"exact dups across splits      {report['exact_duplicates_across_splits']}")
    print(f"sequences spanning splits     {report['sequences_spanning_splits']}")
    print(f"near-dup eval files (RMSE<{args.rmse_threshold})  "
          f"{near_eval_files}/{eval_total} ({report['near_duplicate_eval_pct']}%)")
    print(f"eval->train nearest RMSE      median {report['eval_to_train_min_rmse']['median']}"
          f"  min {report['eval_to_train_min_rmse']['min']}")
    print(f"unrelated train-pair RMSE     median "
          f"{report['unrelated_train_pair_rmse']['median']}  (baseline for comparison)")
    print(f"LEAKAGE FREE: {report['leakage_free']}")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
