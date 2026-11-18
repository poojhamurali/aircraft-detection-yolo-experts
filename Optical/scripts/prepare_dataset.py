"""Preprocess the HRPlanesv2 optical dataset into trainable YOLO layouts.

Source: ../dataset  (2,120 Google Earth airport scenes, 1 class: plane)

The split shipped in ../dataset/data.yaml cannot measure generalisation:

  * it declares `val: images/test`, and images/val is empty, so the model is
    selected on the same 424 images it is scored on;
  * every one of those 424 test images comes from an airport that also appears
    in training, so the detector is asked about runways, terminals, aircraft
    liveries and imaging conditions it has already memorised;
  * 4 images are byte-identical across train and test, and 18 more are visually
    near-identical.

The thesis proposal promises *airport-disjoint* evaluation. This script delivers
it. Filenames encode the source airport (AMST_0031, SZX_0102, GBG_736, plus the
unnamed DT1..DT8 and set1/set2 collections), so whole airports can be assigned to
one split.

Two variants are produced:

  random_split   deduplicated, split by image at 70/15/15 with airports mixed
                 across all three. Kept only so the cost of airport overlap can
                 be measured rather than asserted.
  airport_disjoint  deduplicated, split so that no airport contributes images to
                 more than one split. This is the one to train on.

Usage:
    python scripts/prepare_dataset.py [--src ../dataset]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["plane"]
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
NEAR_DUP_RMSE = 2.5
THUMB = 64

# 000_DT8.jpg / 012_set2.jpg  -> the trailing token is the collection
INDEXED_RE = re.compile(r"^\d+_(?P<src>DT\d+|set\d+)$", re.IGNORECASE)
# AMST_0031.jpg / SZX_0102.jpg / GBG_736 (2).jpg -> leading letters are the airport
PREFIX_RE = re.compile(r"^(?P<src>[A-Za-z]+?)_?\d")


def source_of(stem: str) -> str:
    """Airport or collection a scene came from.

    HRPlanesv2 mixes two naming conventions. Both are handled here rather than in
    the caller, because getting this wrong silently produces a split that looks
    airport-disjoint and is not.
    """
    m = INDEXED_RE.match(stem)
    if m:
        return m.group("src").upper()
    m = PREFIX_RE.match(stem)
    if m:
        return m.group("src").upper()
    return f"OTHER_{stem}"


def read_labels(path: Path):
    """Return (boxes, problems). Boxes are (cls, xc, yc, w, h) clamped to [0, 1]."""
    boxes, problems = [], []
    if not path.exists():
        return boxes, ["missing label file"]
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"line {lineno}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:])
        except ValueError:
            problems.append(f"line {lineno}: non-numeric field")
            continue
        if not 0 <= cls < len(CLASS_NAMES):
            problems.append(f"line {lineno}: class id {cls} out of range")
            continue
        if w <= 0 or h <= 0:
            problems.append(f"line {lineno}: non-positive box size")
            continue
        if xc - w / 2 < -1e-6 or xc + w / 2 > 1 + 1e-6 or yc - h / 2 < -1e-6 or yc + h / 2 > 1 + 1e-6:
            problems.append(f"line {lineno}: box crosses image border (clamped)")
            x1, y1 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
            x2, y2 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
            xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        boxes.append((cls, xc, yc, w, h))
    return boxes, problems


def file_digest(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_source(src: Path):
    records = []
    for split in ("train", "val", "test"):
        img_dir, lbl_dir = src / "images" / split, src / "labels" / split
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.glob("*.jpg")):
            boxes, problems = read_labels(lbl_dir / f"{img.stem}.txt")
            records.append({
                "image": img,
                "stem": img.stem,
                "source": source_of(img.stem),
                "shipped_split": split,
                "boxes": boxes,
                "problems": problems,
                "digest": file_digest(img),
            })
    if not records:
        raise SystemExit(f"no images found under {src}")
    return records


def thumbnails(records, size=THUMB):
    import cv2
    import numpy as np
    rows = []
    for r in records:
        img = cv2.imread(str(r["image"]), cv2.IMREAD_GRAYSCALE)
        rows.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                    .astype(np.float64).ravel())
    return np.stack(rows)


def audit(records, deduped):
    """Everything wrong with the split as shipped, quantified."""
    import numpy as np

    by_digest = defaultdict(list)
    for r in records:
        by_digest[r["digest"]].append(r)
    cross = {d: rs for d, rs in by_digest.items()
             if len({x["shipped_split"] for x in rs}) > 1}

    shipped_sources = defaultdict(set)
    for r in records:
        shipped_sources[r["source"]].add(r["shipped_split"])
    shared = {s for s, v in shipped_sources.items() if len(v) > 1}
    test_from_shared = sum(1 for r in records
                           if r["shipped_split"] == "test" and r["source"] in shared)
    n_test = sum(1 for r in records if r["shipped_split"] == "test")

    # near-duplicates between shipped train and shipped test
    train = [r for r in deduped if r["shipped_split"] == "train"]
    test = [r for r in deduped if r["shipped_split"] == "test"]
    near = None
    if train and test:
        A, B = thumbnails(train), thumbnails(test)
        asq, bsq = (A * A).sum(1), (B * B).sum(1)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            gram = B @ A.T
        rmse = np.sqrt(np.maximum(bsq[:, None] + asq[None, :] - 2 * gram, 0) / A.shape[1])
        mn = rmse.min(axis=1)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(A), size=(400, 2))
        diff = A[idx[:, 0]] - A[idx[:, 1]]
        baseline = np.sqrt((diff * diff).sum(1) / A.shape[1])
        near = {
            "test_images_with_train_twin": int((mn < NEAR_DUP_RMSE).sum()),
            "test_images": len(test),
            "pct": round(100.0 * float((mn < NEAR_DUP_RMSE).mean()), 1),
            "nearest_rmse_median": round(float(np.median(mn)), 1),
            "unrelated_pair_rmse_median": round(float(np.median(baseline)), 1),
        }

    # Box sizes need the real frame dimensions. HRPlanesv2 scenes are 4800x2703,
    # not the 640x640 of the infrared set, so assuming 640 here understates every
    # box by roughly 7x. Dimensions come from the JPEG header rather than a full
    # decode, which matters across 2120 scenes of this size.
    #
    # Two scales are reported. Native pixels say how large the aircraft really
    # are; the second rescales each box to the 640 px input the model is actually
    # shown, which is the number that governs whether a target is detectable and
    # the only one comparable with the infrared stream.
    from PIL import Image

    dims = {}
    sizes_native, sides_at_input = [], []
    for r in records:
        path = r["image"]
        if path not in dims:
            with Image.open(path) as im:
                dims[path] = im.size  # (w, h), header only
        iw, ih = dims[path]
        scale = 640.0 / max(iw, ih)  # letterboxed: long side maps to 640
        for _, _, _, w, h in r["boxes"]:
            bw, bh = w * iw, h * ih
            sizes_native.append(bw * bh)
            sides_at_input.append(((bw * scale) * (bh * scale)) ** 0.5)
    sizes = sorted(sizes_native)
    sides_at_input.sort()

    def pct_under(vals, t):
        return round(100.0 * sum(1 for v in vals if v < t) / len(vals), 1) if vals else 0.0

    frame_sizes = Counter(dims.values())

    return {
        "images": len(records),
        "unique_images_by_content": len(by_digest),
        "duplicate_image_files": len(records) - len(by_digest),
        "duplicate_groups_crossing_shipped_splits": len(cross),
        "boxes": sum(len(r["boxes"]) for r in records),
        "empty_images": sum(1 for r in records if not r["boxes"]),
        "label_problem_count": sum(1 for r in records if r["problems"]),
        "sources": len(shipped_sources),
        "sources_in_more_than_one_shipped_split": len(shared),
        "shipped_test_images_from_a_training_airport": test_from_shared,
        "shipped_test_images": n_test,
        "shipped_test_airport_overlap_pct": round(100.0 * test_from_shared / n_test, 1)
                                            if n_test else 0.0,
        "shipped_val_images": sum(1 for r in records if r["shipped_split"] == "val"),
        "near_duplicates_shipped": near,
        "frame_size_most_common": list(frame_sizes.most_common(1)[0][0]),
        "frame_sizes_distinct": len(frame_sizes),
        "median_box_area_px": round(sizes[len(sizes) // 2], 1) if sizes else 0,
        "median_box_side_px": round(sizes[len(sizes) // 2] ** 0.5, 1) if sizes else 0,
        "median_box_side_at_640_input_px": round(
            sides_at_input[len(sides_at_input) // 2], 1) if sides_at_input else 0,
        "pct_below_32x32_at_640_input": pct_under(sides_at_input, 32),
        "pct_below_16x16_at_640_input": pct_under(sides_at_input, 16),
    }


def dedupe(records):
    by_digest = defaultdict(list)
    for r in records:
        by_digest[r["digest"]].append(r)
    kept = [sorted(rs, key=lambda r: r["stem"])[0] for rs in by_digest.values()]
    return sorted(kept, key=lambda r: r["stem"])


def airport_disjoint_split(records, field="clean_split", seed=0):
    """Assign whole airports to one split, balancing image counts.

    Group sizes here are extremely uneven -- one airport holds nearly 30% of the
    dataset -- so proportional assignment by *group count* would produce wildly
    wrong image proportions. Groups are therefore placed largest-first into
    whichever split is furthest below its image quota, which keeps the realised
    image split close to 70/15/15 without ever splitting an airport.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in records:
        groups[r["source"]].append(r)

    total = len(records)
    quota = {s: SPLIT_FRACTIONS[s] * total for s in SPLIT_FRACTIONS}
    filled = {s: 0 for s in SPLIT_FRACTIONS}

    order = sorted(groups, key=lambda g: (-len(groups[g]), g))
    # break ties between equally sized groups reproducibly but not alphabetically,
    # so that val/test are not dominated by one part of the alphabet
    rng.shuffle(order)
    order.sort(key=lambda g: -len(groups[g]))

    assignment = {}
    for g in order:
        target = max(quota, key=lambda s: (quota[s] - filled[s]) / max(quota[s], 1))
        assignment[g] = target
        filled[target] += len(groups[g])

    for r in records:
        r[field] = assignment[r["source"]]
    return assignment


def random_split(records, field="random_split", seed=0):
    """Image-level split that deliberately mixes airports, for comparison only."""
    rng = random.Random(seed)
    shuffled = sorted(records, key=lambda r: r["stem"])
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * SPLIT_FRACTIONS["train"])
    n_val = round(n * SPLIT_FRACTIONS["val"])
    for i, r in enumerate(shuffled):
        r[field] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")


def materialise(records, out_dir: Path, split_key: str):
    """Write a YOLO dataset tree. Images are symlinked; the source stays read-only."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = Counter()
    for r in records:
        split = r[split_key]
        dst_img = out_dir / "images" / split / f"{r['stem']}.jpg"
        dst_lbl = out_dir / "labels" / split / f"{r['stem']}.txt"
        os.symlink(os.path.relpath(r["image"], dst_img.parent), dst_img)
        dst_lbl.write_text("".join(
            f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for c, xc, yc, w, h in r["boxes"]))
        counts[split] += 1

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"# HRPlanesv2 optical - generated by Optical/scripts/prepare_dataset.py\n"
        f"path: {out_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    return counts, yaml_path


def split_summary(records, split_key):
    rows = {}
    for split in ("train", "val", "test"):
        subset = [r for r in records if r[split_key] == split]
        rows[split] = {
            "images": len(subset),
            "boxes": sum(len(r["boxes"]) for r in subset),
            "airports": sorted({r["source"] for r in subset}),
            "airport_count": len({r["source"] for r in subset}),
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE.parent / "dataset"))
    ap.add_argument("--out", default=str(HERE / "datasets"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit-only", action="store_true",
                    help="refresh outputs/dataset_report.json without rebuilding "
                         "datasets/, so the splits a trained model was scored on "
                         "are left exactly as they are")
    args = ap.parse_args()

    src, out = Path(args.src).resolve(), Path(args.out)
    print(f"scanning {src} ...")
    records = scan_source(src)
    deduped = dedupe(records)
    report = audit(records, deduped)

    print(f"  images {report['images']}  boxes {report['boxes']}  "
          f"empty {report['empty_images']}")
    print(f"  unique images by content: {report['unique_images_by_content']} "
          f"({report['duplicate_image_files']} duplicate files, "
          f"{report['duplicate_groups_crossing_shipped_splits']} cross train/test)")
    print(f"  airports/collections: {report['sources']}")
    print(f"  shipped val images: {report['shipped_val_images']}  "
          f"(data.yaml points val at the test folder)")
    print(f"  shipped test images from an airport also in train: "
          f"{report['shipped_test_images_from_a_training_airport']}"
          f"/{report['shipped_test_images']} "
          f"({report['shipped_test_airport_overlap_pct']}%)")
    if report["near_duplicates_shipped"]:
        nd = report["near_duplicates_shipped"]
        print(f"  near-duplicate test images: {nd['test_images_with_train_twin']}"
              f"/{nd['test_images']} ({nd['pct']}%) at RMSE < {NEAR_DUP_RMSE}; "
              f"unrelated pairs sit at {nd['unrelated_pair_rmse_median']}")

    random_split(deduped, "random_split", seed=args.seed)
    airport_disjoint_split(deduped, "clean_split", seed=args.seed)

    if args.audit_only:
        print("\n--audit-only: datasets/ left untouched")
    else:
        counts_r, yaml_r = materialise(deduped, out / "random_split", "random_split")
        print(f"\nrandom_split      -> {yaml_r}  {dict(counts_r)}  (airports mixed; comparison only)")

        counts_c, yaml_c = materialise(deduped, out / "airport_disjoint", "clean_split")
        print(f"airport_disjoint  -> {yaml_c}  {dict(counts_c)}")

    summary = split_summary(deduped, "clean_split")
    for s in ("train", "val", "test"):
        print(f"  {s:<6} {summary[s]['images']:>5} images  "
              f"{summary[s]['boxes']:>6} boxes  "
              f"{summary[s]['airport_count']:>2} airports: "
              f"{', '.join(summary[s]['airports'][:8])}"
              f"{' ...' if summary[s]['airport_count'] > 8 else ''}")

    report["random_split_summary"] = split_summary(deduped, "random_split")
    report["airport_disjoint_split_summary"] = summary

    out_path = HERE / "outputs" / "dataset_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport -> {out_path}")


if __name__ == "__main__":
    main()
