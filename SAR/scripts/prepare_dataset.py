"""Build the SAR aircraft dataset from SARDet-100K, and audit it for leakage.

SARDet-100K merges ten SAR datasets into one COCO-format benchmark with six
classes. We want one of them, so this filters to `aircraft` and converts to YOLO
format.

Two things about this data drive the design.

First, provenance is gone. SARDet-100K renumbers every file (`0086537.jpg`), so
nothing in the name says which source dataset or which original scene a patch
came from. What survives is the frame size, which happens to identify the source
exactly: 512x512 is SAR-AIRcraft (Gaofen-3, 1 m), 224x224 is SADD (TerraSAR-X)
and 256x256 is MSAR. Those counts reproduce the paper's table.

Second, and the reason this script exists at all: SAR-AIRcraft patches are crops
taken from a small number of large Gaofen-3 scenes. Neighbouring crops overlap.
If two overlapping crops land on opposite sides of a split, the model is tested
on ground it memorised, which is exactly what cost 0.19 mAP on the infrared
stream. Filenames cannot reveal this, so it has to be measured from the pixels.

Outputs two variants:

  official  the split as SARDet-100K ships it, kept for comparison only
  clean     no near-duplicate group spans two splits; quote this one

    ../.venv/bin/python scripts/prepare_dataset.py
    ../.venv/bin/python scripts/prepare_dataset.py --audit-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "raw"
AIRCRAFT_CATEGORY = 1
CLASS_NAMES = ["aircraft"]

# Frame size identifies the source dataset; see module docstring.
SOURCE_BY_SIZE = {(512, 512): "SAR-AIRcraft", (224, 224): "SADD", (256, 256): "MSAR"}

# Chosen from scripts/calibrate_threshold.py rather than guessed. On this data
# the nearest-neighbour distance is 0.00 for over half the images, and the
# fraction flagged barely moves between a threshold of 1 and 10 (62.5% -> 74.3%),
# so the duplicates are genuinely identical rather than marginal. At 2.0 only
# 0.02% of unrelated pairs fall below, which makes false positives negligible.
NEAR_DUP_RMSE = 2.0
THUMB = 64


def source_of(w, h):
    return SOURCE_BY_SIZE.get((w, h), f"other-{w}x{h}")


def scan(splits=("train", "val", "test")):
    """COCO annotations -> one record per image that contains an aircraft."""
    records = []
    for split in splits:
        ann_path = RAW / "Annotations" / f"{split}.json"
        img_dir = RAW / "extracted" / split
        if not ann_path.exists() or not img_dir.is_dir():
            print(f"  skipping {split}: not downloaded yet")
            continue
        data = json.loads(ann_path.read_text())
        by_id = {i["id"]: i for i in data["images"]}
        boxes = defaultdict(list)
        for a in data["annotations"]:
            if a["category_id"] != AIRCRAFT_CATEGORY:
                continue
            if a.get("iscrowd"):
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes[a["image_id"]].append((x, y, w, h))

        for img_id, bs in boxes.items():
            info = by_id[img_id]
            path = img_dir / info["file_name"]
            if not path.exists():
                continue
            records.append({
                "path": path,
                "name": info["file_name"],
                "official_split": split,
                "width": info["width"],
                "height": info["height"],
                "source": source_of(info["width"], info["height"]),
                "boxes": bs,
            })
    return records


def to_yolo(rec):
    """COCO xywh in pixels -> YOLO cx cy w h normalised, clipped to the frame."""
    W, H = rec["width"], rec["height"]
    lines = []
    for x, y, w, h in rec["boxes"]:
        x1, y1 = max(x, 0.0), max(y, 0.0)
        x2, y2 = min(x + w, W), min(y + h, H)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"0 {((x1 + x2) / 2) / W:.6f} {((y1 + y2) / 2) / H:.6f} "
                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def thumbnails(records, size=THUMB):
    """Flattened grey thumbnails, one row per record."""
    X = np.zeros((len(records), size * size), dtype=np.float32)
    for i, r in enumerate(records):
        img = cv2.imread(str(r["path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        X[i] = cv2.resize(img, (size, size),
                          interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return X


class Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def visual_groups(records, threshold=NEAR_DUP_RMSE, block=2048):
    """Union near-identical images into groups, and label each record.

    Pairwise RMSE via the Gram trick, ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b, in
    blocks: the full 20k x 20k matrix would be 1.6 GB, while one block row is a
    few hundred MB and the result is identical.

    A perceptual hash is deliberately not used. It failed on the infrared set,
    where near-uniform frames collapsed into a handful of buckets and flagged
    unrelated pairs as duplicates. SAR patches are likewise dominated by flat
    speckle, so pixel RMSE against a measured unrelated-pair baseline is the
    trustworthy comparison.
    """
    X = thumbnails(records)
    n = len(records)
    sq = (X * X).sum(axis=1)
    uf = Union(n)
    pairs = 0

    for start in range(0, n, block):
        stop = min(start + block, n)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            gram = X[start:stop] @ X.T
        d2 = sq[start:stop, None] + sq[None, :] - 2.0 * gram
        rmse = np.sqrt(np.maximum(d2, 0.0) / X.shape[1])
        for local in range(stop - start):
            i = start + local
            rmse[local, i] = np.inf
            hits = np.nonzero(rmse[local] < threshold)[0]
            for j in hits:
                if j != i:
                    uf.union(i, int(j))
                    pairs += 1

    for i, r in enumerate(records):
        r["group"] = uf.find(i)

    # An unrelated-pair baseline says whether the threshold is meaningful at all.
    rng = np.random.default_rng(0)
    a = rng.integers(0, n, size=min(4000, n))
    b = rng.integers(0, n, size=min(4000, n))
    keep = a != b
    base = np.sqrt(((X[a[keep]] - X[b[keep]]) ** 2).mean(axis=1))

    return {
        "groups": len({r["group"] for r in records}),
        "near_duplicate_pairs": pairs // 2,
        "threshold": threshold,
        "unrelated_pair_rmse_median": round(float(np.median(base)), 1),
        "unrelated_pair_rmse_p1": round(float(np.percentile(base, 1)), 1),
    }


def group_split(records, field, seed=0, ratios=(0.70, 0.15, 0.15)):
    """Assign whole groups to splits, largest first so the sizes land close."""
    by_group = defaultdict(list)
    for r in records:
        by_group[r["group"]].append(r)

    groups = sorted(by_group.values(), key=len, reverse=True)
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    total = len(records)
    targets = [ratios[0] * total, ratios[1] * total, ratios[2] * total]
    names = ["train", "val", "test"]
    filled = [0, 0, 0]

    for grp in groups:
        i = max(range(3), key=lambda k: targets[k] - filled[k])
        for r in grp:
            r[field] = names[i]
        filled[i] += len(grp)
    return dict(zip(names, filled))


def audit(records, group_stats):
    by_digest = defaultdict(list)
    for r in records:
        by_digest[md5(r["path"])].append(r)

    cross = 0
    for group in by_digest.values():
        if len({r["official_split"] for r in group}) > 1:
            cross += 1

    # How much of the shipped eval set has a near-identical twin in shipped train
    train_groups = {r["group"] for r in records if r["official_split"] == "train"}
    evals = [r for r in records if r["official_split"] in ("val", "test")]
    leaked = sum(1 for r in evals if r["group"] in train_groups)

    sides = []
    for r in records:
        for _, _, w, h in r["boxes"]:
            sides.append((w * h) ** 0.5)
    sides.sort()

    def pct_under(t):
        return round(100.0 * sum(1 for s in sides if s < t) / len(sides), 1) if sides else 0.0

    return {
        "images_with_aircraft": len(records),
        "aircraft_boxes": sum(len(r["boxes"]) for r in records),
        "unique_images_by_content": len(by_digest),
        "duplicate_image_files": len(records) - len(by_digest),
        "duplicate_groups_crossing_official_splits": cross,
        "sources": dict(Counter(r["source"] for r in records)),
        "official_split_sizes": dict(Counter(r["official_split"] for r in records)),
        "visual_groups": group_stats,
        "official_eval_images": len(evals),
        "official_eval_images_with_train_twin": leaked,
        "official_eval_leak_pct": round(100.0 * leaked / len(evals), 1) if evals else 0.0,
        "median_box_side_px": round(sides[len(sides) // 2], 1) if sides else 0,
        "pct_boxes_below_32x32": pct_under(32),
        "pct_boxes_below_16x16": pct_under(16),
    }


def materialise(records, out_dir, field):
    """Symlink images and write YOLO labels. Source images are never modified."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    counts = Counter()
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for r in records:
        split = r.get(field)
        if split is None:
            continue
        lines = to_yolo(r)
        if not lines:
            continue
        stem = Path(r["name"]).stem
        link = out_dir / "images" / split / f"{stem}.jpg"
        if not link.exists():
            link.symlink_to(r["path"].resolve())
        (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        counts[split] += 1

    yaml = out_dir / "data.yaml"
    yaml.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")
    return counts, yaml


def split_summary(records, field):
    out = {}
    for split in ("train", "val", "test"):
        rs = [r for r in records if r.get(field) == split]
        out[split] = {
            "images": len(rs),
            "boxes": sum(len(r["boxes"]) for r in rs),
            "sources": dict(Counter(r["source"] for r in rs)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "datasets"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    print("scanning annotations ...")
    records = scan()
    if not records:
        raise SystemExit("no images found; run scripts/download.py first")
    print(f"  {len(records)} images with aircraft, "
          f"{sum(len(r['boxes']) for r in records)} boxes")
    print(f"  sources: {dict(Counter(r['source'] for r in records))}")

    print("grouping near-duplicates (this reads every image) ...")
    gstats = visual_groups(records)
    print(f"  {gstats['groups']} visual groups from {len(records)} images; "
          f"{gstats['near_duplicate_pairs']} near-duplicate pairs at RMSE < "
          f"{gstats['threshold']}")
    print(f"  unrelated pairs sit at RMSE {gstats['unrelated_pair_rmse_median']} "
          f"(1st percentile {gstats['unrelated_pair_rmse_p1']})")

    report = audit(records, gstats)
    print(f"  official split leaks {report['official_eval_images_with_train_twin']}"
          f"/{report['official_eval_images']} eval images "
          f"({report['official_eval_leak_pct']}%) as near-duplicates of train")

    for r in records:
        r["official"] = r["official_split"]
    group_split(records, "clean", seed=args.seed)

    if args.audit_only:
        print("\n--audit-only: datasets/ left untouched")
    else:
        out = Path(args.out)
        for name, field in (("official", "official"), ("clean", "clean")):
            counts, yaml = materialise(records, out / name, field)
            print(f"\n{name:<9} -> {yaml}  {dict(counts)}")

    report["official_summary"] = split_summary(records, "official")
    report["clean_summary"] = split_summary(records, "clean")

    dest = HERE / "outputs" / "dataset_report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport -> {dest}")


if __name__ == "__main__":
    main()
