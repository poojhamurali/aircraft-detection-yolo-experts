"""Preprocess the IRdata-4465 infrared dataset into trainable YOLO layouts.

Source: ~/Downloads/data  (4,465 infrared frames, 640x640, YOLO horizontal boxes,
4 classes: airplane / bird / drone / helicopter)

Three dataset variants are produced:

  official      the split exactly as shipped, symlinked. Kept only as a baseline
                so the effect of its contamination can be measured.
  seq_disjoint  byte-identical duplicate frames removed, then split at the level
                of the source video sequence (IR_DRONE_045, IR_AIRPLANE_013, ...)
                so no sequence contributes frames to more than one split.
  clean         as seq_disjoint, but the split unit is a *visual* group rather
                than a sequence id. This is the variant to train on.

seq_disjoint is not actually leakage-free, which is why `clean` exists. The
dataset author cut single source videos into consecutively numbered sequences, so
IR_HELICOPTER_018 and IR_HELICOPTER_019 are the same footage under two ids; a
sequence-level split happily puts one in train and the other in test. Measured on
seq_disjoint, 153 of 559 eval frames (27%) still had a visually near-identical
twin in train at thumbnail RMSE < 1, against a median of 61 for unrelated frames.

`clean` therefore unions two relations before splitting: same sequence id, and
thumbnail RMSE below --near-dup-rmse. Connected components of that union are the
split unit. The threshold matters: at 2.5 the dataset forms 241 groups with the
largest holding 4% of frames, while at 3.0 transitive chaining collapses 12% of
the data into one component and the split stops being balanced.

Frames named IR_frame_<n> carry no sequence id, so they are grouped into
pseudo-sequences by frame-number bin to keep temporally adjacent frames together.

Usage:
    python scripts/prepare_dataset.py [--src PATH] [--clahe]
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

CLASS_NAMES = ["airplane", "bird", "drone", "helicopter"]
NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
IMG_SUFFIX = ".jpg"
FRAME_BIN = 100  # pseudo-sequence width for unnamed IR_frame_<n> images
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

HERE = Path(__file__).resolve().parent.parent
COPY_RE = re.compile(r"^(?P<base>.+?)_jpg\.rf\.(?P<hash>[0-9a-f]+)$")
SEQ_RE = re.compile(r"^(?P<seq>IR_(?P<cls>[A-Z]+)_\d+)_frame_(?P<frame>\d+)$")
PLAIN_RE = re.compile(r"^IR_frame_(?P<frame>\d+)$")


# --------------------------------------------------------------------------
# parsing helpers
def base_frame(stem: str) -> str:
    """IR_DRONE_045_frame_125_jpg.rf.<hash>  ->  IR_DRONE_045_frame_125"""
    m = COPY_RE.match(stem)
    return m.group("base") if m else stem


def sequence_of(base: str) -> str:
    """Source video a frame came from, or a frame-number bin when unnamed."""
    m = SEQ_RE.match(base)
    if m:
        return m.group("seq")
    m = PLAIN_RE.match(base)
    if m:
        return f"IR_UNNAMED_bin{int(m.group('frame')) // FRAME_BIN:02d}"
    return f"IR_OTHER_{base}"


def class_from_name(base: str) -> str | None:
    m = SEQ_RE.match(base)
    return m.group("cls").lower() if m else None


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
        if not all(0.0 <= v <= 1.0 for v in (xc, yc)) or w <= 0 or h <= 0:
            problems.append(f"line {lineno}: coordinates out of range")
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


# --------------------------------------------------------------------------
# scanning
def scan_source(src: Path):
    """Collect one record per image file in the shipped dataset."""
    records = []
    for split in ("train", "val", "test"):
        img_dir, lbl_dir = src / "images" / split, src / "labels" / split
        if not img_dir.is_dir():
            raise SystemExit(f"missing source directory: {img_dir}")
        for img in sorted(img_dir.glob(f"*{IMG_SUFFIX}")):
            base = base_frame(img.stem)
            boxes, problems = read_labels(lbl_dir / f"{img.stem}.txt")
            records.append({
                "image": img,
                "label": lbl_dir / f"{img.stem}.txt",
                "stem": img.stem,
                "base": base,
                "sequence": sequence_of(base),
                "name_class": class_from_name(base),
                "official_split": split,
                "boxes": boxes,
                "problems": problems,
                "digest": file_digest(img),
            })
    return records


def audit(records):
    """Label sanity, name-vs-id agreement, duplicate and leakage accounting."""
    cls_counter = Counter()
    box_px = []
    problems = defaultdict(list)
    empty = []
    mismatched_name = []

    for r in records:
        if r["problems"]:
            problems[r["stem"]] = r["problems"]
        if not r["boxes"]:
            empty.append(r["stem"])
        for cls, _, _, w, h in r["boxes"]:
            cls_counter[cls] += 1
            box_px.append((w * 640, h * 640))
        if r["name_class"] and r["boxes"]:
            ids = {b[0] for b in r["boxes"]}
            expected = NAME_TO_ID.get(r["name_class"])
            if expected is not None and ids != {expected}:
                mismatched_name.append((r["stem"], sorted(ids), expected))

    by_digest = defaultdict(list)
    for r in records:
        by_digest[r["digest"]].append(r)
    dup_groups = {d: rs for d, rs in by_digest.items() if len(rs) > 1}
    cross_split_dups = {
        d: rs for d, rs in dup_groups.items()
        if len({r["official_split"] for r in rs}) > 1
    }
    contaminated_eval_files = sum(
        1 for rs in cross_split_dups.values()
        for r in rs if r["official_split"] in ("val", "test")
    )

    seq_splits = defaultdict(set)
    for r in records:
        seq_splits[r["sequence"]].add(r["official_split"])
    seqs_spanning_splits = sum(1 for v in seq_splits.values() if len(v) > 1)

    areas = [w * h for w, h in box_px]
    areas.sort()

    def pct(threshold):
        return 100.0 * sum(1 for a in areas if a < threshold) / len(areas) if areas else 0.0

    return {
        "images": len(records),
        "boxes": sum(cls_counter.values()),
        "empty_images": len(empty),
        "label_problems": {k: v for k, v in list(problems.items())[:20]},
        "label_problem_count": len(problems),
        "class_counts": {CLASS_NAMES[c]: n for c, n in sorted(cls_counter.items())},
        "filename_class_mismatches": mismatched_name[:20],
        "filename_class_mismatch_count": len(mismatched_name),
        "unique_images_by_content": len(by_digest),
        "duplicate_image_files": len(records) - len(by_digest),
        "duplicate_groups": len(dup_groups),
        "duplicate_groups_crossing_splits": len(cross_split_dups),
        "contaminated_val_test_files": contaminated_eval_files,
        "sequences": len(seq_splits),
        "sequences_spanning_official_splits": seqs_spanning_splits,
        "box_px": {
            "median_area": areas[len(areas) // 2] if areas else 0,
            "pct_below_32x32": round(pct(32 * 32), 1),
            "pct_below_16x16": round(pct(16 * 16), 1),
        },
    }


# --------------------------------------------------------------------------
# split construction
def dedupe(records):
    """Keep one record per unique image content, preferring a deterministic stem."""
    by_digest = defaultdict(list)
    for r in records:
        by_digest[r["digest"]].append(r)
    kept = [sorted(rs, key=lambda r: r["stem"])[0] for rs in by_digest.values()]
    return sorted(kept, key=lambda r: r["stem"])


def group_split(records, group_key, split_field, seed=0):
    """Assign whole groups to train/val/test, balanced per dominant class."""
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for r in records:
        grouped[r[group_key]].append(r)

    by_class = defaultdict(list)
    for key, rs in grouped.items():
        counts = Counter(b[0] for r in rs for b in r["boxes"])
        dominant = counts.most_common(1)[0][0] if counts else -1
        by_class[dominant].append(key)

    assignment = {}
    for dominant, keys in sorted(by_class.items()):
        keys = sorted(keys, key=str)
        rng.shuffle(keys)
        n = len(keys)
        n_train = round(n * SPLIT_FRACTIONS["train"])
        n_val = round(n * SPLIT_FRACTIONS["val"])
        # every class group must still contribute to val and test when possible
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        for i, key in enumerate(keys):
            if i < n_train:
                assignment[key] = "train"
            elif i < n_train + n_val:
                assignment[key] = "val"
            else:
                assignment[key] = "test"

    for r in records:
        r[split_field] = assignment[r[group_key]]
    return assignment


def thumbnails(records, size=64):
    """Flattened grayscale thumbnails, one row per record."""
    import cv2
    import numpy as np
    rows = []
    for r in records:
        img = cv2.imread(str(r["image"]), cv2.IMREAD_GRAYSCALE)
        rows.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                    .astype(np.float64).ravel())
    return np.stack(rows)


def visual_groups(records, rmse_threshold=2.5, thumb=64):
    """Union sequence identity with visual near-duplication; label each record.

    Two frames from *different* sequence ids can be the same footage, because the
    source videos were cut into consecutively numbered sequences. Splitting on
    sequence id alone therefore leaks. Connected components of
    (same sequence) OR (thumbnail RMSE < threshold) do not.
    """
    import numpy as np

    n = len(records)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_seq = defaultdict(list)
    for i, r in enumerate(records):
        by_seq[r["sequence"]].append(i)
    for idxs in by_seq.values():
        for i in idxs[1:]:
            union(idxs[0], i)

    X = thumbnails(records, thumb)
    sq = (X * X).sum(axis=1)
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b, then per-pixel RMSE.
    # errstate: numpy's Accelerate BLAS backend raises spurious divide/overflow
    # warnings from this gemm on macOS; the result is correct.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        gram = X @ X.T
    rmse = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0) / X.shape[1])
    np.fill_diagonal(rmse, np.inf)

    near_i, near_j = np.where(np.triu(rmse < rmse_threshold, 1))
    for i, j in zip(near_i, near_j):
        union(int(i), int(j))

    for i, r in enumerate(records):
        r["group"] = f"g{find(i):04d}"

    sizes = Counter(r["group"] for r in records)
    cross_seq_links = sum(
        1 for i, j in zip(near_i, near_j)
        if records[int(i)]["sequence"] != records[int(j)]["sequence"])
    return {
        "near_dup_rmse_threshold": rmse_threshold,
        "groups": len(sizes),
        "largest_group_images": max(sizes.values()),
        "largest_group_pct": round(100.0 * max(sizes.values()) / n, 1),
        "near_duplicate_pairs": int(len(near_i)),
        "near_duplicate_pairs_across_sequences": cross_seq_links,
        "sequences_merged_by_visual_similarity": len(by_seq) - len(sizes),
    }


# --------------------------------------------------------------------------
# materialisation
def materialise(records, out_dir: Path, split_key: str, clahe=False):
    """Write a YOLO dataset tree. Images are symlinked unless CLAHE is applied."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    if clahe:
        import cv2
        enhancer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    counts = Counter()
    for r in records:
        split = r[split_key]
        dst_img = out_dir / "images" / split / f"{r['stem']}{IMG_SUFFIX}"
        dst_lbl = out_dir / "labels" / split / f"{r['stem']}.txt"

        if clahe:
            import cv2
            gray = cv2.imread(str(r["image"]), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(dst_img), cv2.cvtColor(enhancer.apply(gray), cv2.COLOR_GRAY2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            os.symlink(os.path.relpath(r["image"], dst_img.parent), dst_img)

        dst_lbl.write_text("".join(
            f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for c, xc, yc, w, h in r["boxes"]
        ))
        counts[split] += 1

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"# IRdata-4465 infrared aircraft detection - generated by prepare_dataset.py\n"
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
        cls = Counter(b[0] for r in subset for b in r["boxes"])
        rows[split] = {
            "images": len(subset),
            "boxes": sum(cls.values()),
            "sequences": len({r["sequence"] for r in subset}),
            "per_class": {CLASS_NAMES[c]: cls.get(c, 0) for c in range(len(CLASS_NAMES))},
        }
        if all("group" in r for r in subset):
            rows[split]["groups"] = len({r["group"] for r in subset})
    return rows


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "Downloads" / "data"))
    ap.add_argument("--out", default=str(HERE / "datasets"))
    ap.add_argument("--clahe", action="store_true",
                    help="also build a CLAHE contrast-enhanced copy of the clean split")
    ap.add_argument("--near-dup-rmse", type=float, default=2.5,
                    help="thumbnail RMSE below which two frames are the same shot; "
                         "above ~3.0 transitive chaining collapses the dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, out = Path(args.src).expanduser(), Path(args.out)
    print(f"scanning {src} ...")
    records = scan_source(src)
    report = audit(records)

    print(f"  images {report['images']}  boxes {report['boxes']}  "
          f"empty {report['empty_images']}")
    print(f"  label problems: {report['label_problem_count']}")
    print(f"  filename/class-id mismatches: {report['filename_class_mismatch_count']}")
    print(f"  byte-identical duplicate files: {report['duplicate_image_files']} "
          f"({report['unique_images_by_content']} unique images)")
    print(f"  duplicate groups crossing official splits: "
          f"{report['duplicate_groups_crossing_splits']} -> "
          f"{report['contaminated_val_test_files']} contaminated val/test files")
    print(f"  sequences: {report['sequences']} "
          f"({report['sequences_spanning_official_splits']} span official splits)")

    counts_off, yaml_off = materialise(records, out / "official", "official_split")
    print(f"\nofficial split  -> {yaml_off}  {dict(counts_off)}")

    deduped = dedupe(records)
    group_split(deduped, "sequence", "seq_split", seed=args.seed)
    counts_seq, yaml_seq = materialise(deduped, out / "seq_disjoint", "seq_split")
    print(f"seq_disjoint    -> {yaml_seq}  {dict(counts_seq)}  (leaks; superseded)")

    print(f"\ngrouping near-duplicate frames (RMSE < {args.near_dup_rmse}) ...")
    grouping = visual_groups(deduped, rmse_threshold=args.near_dup_rmse)
    print(f"  {grouping['groups']} visual groups; largest holds "
          f"{grouping['largest_group_images']} frames ({grouping['largest_group_pct']}%)")
    print(f"  {grouping['near_duplicate_pairs_across_sequences']} near-duplicate pairs "
          f"spanned different sequence ids")
    print(f"  {grouping['sequences_merged_by_visual_similarity']} sequences absorbed into "
          f"another sequence's group")

    group_split(deduped, "group", "clean_split", seed=args.seed)
    counts_clean, yaml_clean = materialise(deduped, out / "clean", "clean_split")
    print(f"clean split     -> {yaml_clean}  {dict(counts_clean)}")

    report["official_split_summary"] = split_summary(records, "official_split")
    report["seq_disjoint_split_summary"] = split_summary(deduped, "seq_split")
    report["grouping"] = grouping
    report["clean_split_summary"] = split_summary(deduped, "clean_split")

    if args.clahe:
        counts_cl, yaml_cl = materialise(deduped, out / "clean_clahe", "clean_split",
                                         clahe=True)
        print(f"clahe split     -> {yaml_cl}  {dict(counts_cl)}")

    report_path = HERE / "outputs" / "dataset_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def jsonable(o):
        return str(o) if isinstance(o, Path) else o

    report_path.write_text(json.dumps(report, indent=2, default=jsonable))
    print(f"\nreport -> {report_path}")


if __name__ == "__main__":
    main()
