"""Quantify which classes a trained IR detector confuses, and why.

mAP alone cannot distinguish "the model never saw the object" from "the model saw
it and called it the wrong thing". Those need opposite fixes, so this script
separates them: every ground-truth box is matched to a prediction by IoU and
classified as correct, cross-class confusion, or a miss.

Matching is greedy by descending confidence at IoU >= 0.5, which is the same rule
mAP@0.5 uses, so the counts here are consistent with the reported metrics.

Also emits per-class box-size statistics, because in infrared sky imagery the
distinguishing features of an airplane and a bird survive only above a certain
number of pixels on target.

Usage:
    python scripts/confusion_analysis.py --run baseline_yolo12s --split test
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["airplane", "bird", "drone", "helicopter"]
BACKGROUND = "background"


def iou_matrix(a, b):
    """Pairwise IoU between two arrays of xyxy boxes."""
    import numpy as np
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def load_gt(label_path: Path, w: int, h: int):
    """YOLO normalised cxcywh -> (class, xyxy in pixels)."""
    boxes, classes = [], []
    if not label_path.exists():
        return boxes, classes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, xc, yc, bw, bh = int(float(parts[0])), *(float(v) for v in parts[1:])
        boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h,
                      (xc + bw / 2) * w, (yc + bh / 2) * h])
        classes.append(c)
    return boxes, classes


def dataset_root(data_yaml: Path) -> Path:
    for line in data_yaml.read_text().splitlines():
        if line.startswith("path:"):
            return Path(line.split(":", 1)[1].strip())
    return data_yaml.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory name under outputs/runs")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    from ultralytics import YOLO

    run_dir = HERE / "outputs" / "runs" / args.run
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"no trained weights at {weights}")

    metrics = json.loads((run_dir / "metrics.json").read_text())
    root = dataset_root(Path(metrics["data"]))
    img_dir = root / "images" / args.split
    lbl_dir = root / "labels" / args.split
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {img_dir}")

    model = YOLO(str(weights))

    labels = CLASS_NAMES + [BACKGROUND]
    matrix = np.zeros((len(labels), len(labels)), dtype=int)  # rows=truth, cols=prediction
    # box area of every gt box, split by whether it was localised and named correctly
    size_outcome = defaultdict(list)
    gt_sizes = defaultdict(list)

    for batch_start in range(0, len(images), 16):
        batch = images[batch_start:batch_start + 16]
        results = model.predict([str(p) for p in batch], conf=args.conf, iou=args.nms_iou,
                                device=args.device, verbose=False)
        for img_path, res in zip(batch, results):
            h, w = res.orig_shape
            gt_boxes, gt_classes = load_gt(lbl_dir / f"{img_path.stem}.txt", w, h)
            pred_boxes = res.boxes.xyxy.cpu().numpy().tolist()
            pred_classes = [int(c) for c in res.boxes.cls.cpu().numpy()]
            pred_conf = res.boxes.conf.cpu().numpy().tolist()

            order = sorted(range(len(pred_boxes)), key=lambda i: -pred_conf[i])
            ious = iou_matrix(pred_boxes, gt_boxes)
            gt_taken, pred_taken = set(), set()

            for pi in order:
                if len(gt_boxes) == 0:
                    break
                candidates = [gi for gi in range(len(gt_boxes))
                              if gi not in gt_taken and ious[pi, gi] >= args.iou_match]
                if not candidates:
                    continue
                gi = max(candidates, key=lambda g: ious[pi, g])
                gt_taken.add(gi)
                pred_taken.add(pi)
                matrix[gt_classes[gi], pred_classes[pi]] += 1
                area = ((gt_boxes[gi][2] - gt_boxes[gi][0])
                        * (gt_boxes[gi][3] - gt_boxes[gi][1]))
                outcome = "correct" if pred_classes[pi] == gt_classes[gi] else "misclassified"
                size_outcome[(CLASS_NAMES[gt_classes[gi]], outcome)].append(area)

            for gi in range(len(gt_boxes)):
                area = ((gt_boxes[gi][2] - gt_boxes[gi][0])
                        * (gt_boxes[gi][3] - gt_boxes[gi][1]))
                gt_sizes[CLASS_NAMES[gt_classes[gi]]].append(area)
                if gi not in gt_taken:
                    matrix[gt_classes[gi], len(CLASS_NAMES)] += 1  # missed entirely
                    size_outcome[(CLASS_NAMES[gt_classes[gi]], "missed")].append(area)

            for pi in range(len(pred_boxes)):
                if pi not in pred_taken:
                    matrix[len(CLASS_NAMES), pred_classes[pi]] += 1  # false positive

    # ---- derive the numbers a reader actually wants ---------------------
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        total = int(matrix[i, :].sum())
        correct = int(matrix[i, i])
        missed = int(matrix[i, len(CLASS_NAMES)])
        wrong = total - correct - missed
        confused_with = {CLASS_NAMES[j]: int(matrix[i, j])
                         for j in range(len(CLASS_NAMES))
                         if j != i and matrix[i, j] > 0}
        per_class[name] = {
            "gt_boxes": total,
            "correct": correct,
            "misclassified": wrong,
            "missed": missed,
            "correct_pct": round(100.0 * correct / total, 1) if total else 0.0,
            "misclassified_pct": round(100.0 * wrong / total, 1) if total else 0.0,
            "missed_pct": round(100.0 * missed / total, 1) if total else 0.0,
            "confused_with": dict(sorted(confused_with.items(), key=lambda kv: -kv[1])),
            "false_positives": int(matrix[len(CLASS_NAMES), i]),
        }

    def pair_rate(a, b):
        """Percent of class-a ground truth that was localised but named b."""
        ia, ib = CLASS_NAMES.index(a), CLASS_NAMES.index(b)
        total = int(matrix[ia, :].sum())
        return round(100.0 * int(matrix[ia, ib]) / total, 1) if total else 0.0

    def median(xs):
        xs = sorted(xs)
        return round(float(xs[len(xs) // 2]), 1) if xs else 0.0

    size_stats = {}
    for name in CLASS_NAMES:
        areas = gt_sizes[name]
        size_stats[name] = {
            "gt_boxes": len(areas),
            "median_area_px": median(areas),
            "median_side_px": round(median(areas) ** 0.5, 1),
            "pct_below_32x32": round(100.0 * sum(1 for a in areas if a < 32 * 32) / len(areas), 1)
                               if areas else 0.0,
            "median_side_correct_px": round(median(size_outcome[(name, "correct")]) ** 0.5, 1),
            "median_side_misclassified_px": round(
                median(size_outcome[(name, "misclassified")]) ** 0.5, 1),
            "median_side_missed_px": round(median(size_outcome[(name, "missed")]) ** 0.5, 1),
        }

    report = {
        "run": args.run,
        "split": args.split,
        "conf_threshold": args.conf,
        "iou_match_threshold": args.iou_match,
        "labels": labels,
        "matrix_rows_truth_cols_pred": matrix.tolist(),
        "per_class": per_class,
        "airplane_called_bird_pct": pair_rate("airplane", "bird"),
        "bird_called_airplane_pct": pair_rate("bird", "airplane"),
        "size_stats": size_stats,
    }

    out = Path(args.out) if args.out else run_dir / f"confusion_{args.split}.json"
    out.write_text(json.dumps(report, indent=2))

    # ---- console table --------------------------------------------------
    width = max(len(s) for s in labels) + 2
    print(f"\n{args.run}  ({args.split} split, conf={args.conf}, IoU={args.iou_match})")
    print("rows = ground truth, columns = prediction\n")
    print("".ljust(width) + "".join(s.ljust(width) for s in labels))
    for i, name in enumerate(labels):
        row = "".join(str(matrix[i, j]).ljust(width) for j in range(len(labels)))
        print(name.ljust(width) + row)

    print(f"\n{'class':<12}{'GT':>6}{'correct':>9}{'wrong cls':>11}{'missed':>8}   confused with")
    for name, d in per_class.items():
        conf = ", ".join(f"{k} {v}" for k, v in d["confused_with"].items()) or "-"
        print(f"{name:<12}{d['gt_boxes']:>6}{d['correct_pct']:>8.1f}%"
              f"{d['misclassified_pct']:>10.1f}%{d['missed_pct']:>7.1f}%   {conf}")

    print(f"\nairplane labelled bird: {report['airplane_called_bird_pct']}% of airplane GT")
    print(f"bird labelled airplane: {report['bird_called_airplane_pct']}% of bird GT")
    print(f"\n{'class':<12}{'median side':>13}{'<32px':>8}{'ok':>7}{'wrong':>7}{'missed':>8}")
    for name, d in size_stats.items():
        print(f"{name:<12}{d['median_side_px']:>12.1f}px{d['pct_below_32x32']:>7.1f}%"
              f"{d['median_side_correct_px']:>7.1f}{d['median_side_misclassified_px']:>7.1f}"
              f"{d['median_side_missed_px']:>8.1f}")
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
