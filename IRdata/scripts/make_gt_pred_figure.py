"""Build a ground-truth vs prediction figure for the infrared stream.

The earlier montage only burned in predictions, so a viewer could not tell a correct
detection from a miss. Here every panel draws both: ground truth in green, model output in
red, with the IoU-matched verdict in the panel title.

Infrared targets here are ~26 px inside a 640 px frame, which is invisible at slide size,
so each sample is shown twice - the full scene for context and a magnified crop around the
target where the two boxes can actually be compared.

Usage:
    python IRdata/scripts/make_gt_pred_figure.py [--run C_clean_yolov8s] [--conf 0.25]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
FIG = HERE / "outputs" / "figures"
TEST_IMG = HERE / "datasets" / "seq_disjoint" / "images" / "test"
TEST_LBL = HERE / "datasets" / "seq_disjoint" / "labels" / "test"

CLASSES = ["airplane", "bird", "drone", "helicopter"]
GT_COLOR = "#00E64D"
PR_COLOR = "#FF3B30"
IOU_HIT = 0.5


def load_gt(stem, w, h):
    """YOLO normalised cx,cy,w,h -> pixel x1,y1,x2,y2."""
    path = TEST_LBL / f"{stem}.txt"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
        out.append((cls, (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def verdict(gts, preds):
    """Best IoU against a same-class prediction, plus a short human-readable label."""
    if not gts:
        return ("false positive" if preds else "empty frame"), 0.0
    best = 0.0
    for cls, *box in gts:
        for pcls, pbox, _ in preds:
            if pcls == cls:
                best = max(best, iou(box, pbox))
    if best >= IOU_HIT:
        return "correct detection", best
    if preds:
        return "missed (box off target)", best
    return "MISSED - no detection", best


def crop_window(boxes, w, h, min_size=190):
    """Square window containing every box, padded, clipped to the frame."""
    if not boxes:
        return 0, 0, w, h
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1) * 3.5
    size = max(size, min_size)
    half = min(size / 2, w / 2, h / 2)
    cx = min(max(cx, half), w - half)
    cy = min(max(cy, half), h - half)
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def draw(ax, gts, preds, offset=(0, 0), lw=1.8, show_labels=True):
    ox, oy = offset
    for cls, *box in gts:
        x1, y1, x2, y2 = box
        ax.add_patch(mpatches.Rectangle((x1 - ox, y1 - oy), x2 - x1, y2 - y1,
                                        fill=False, edgecolor=GT_COLOR, linewidth=lw))
        if show_labels:
            ax.text(x1 - ox, y1 - oy - 4, f"GT {CLASSES[cls]}", color=GT_COLOR,
                    fontsize=7.5, weight="bold",
                    bbox=dict(facecolor="black", alpha=0.55, pad=0.9, edgecolor="none"))
    for cls, box, conf in preds:
        x1, y1, x2, y2 = box
        ax.add_patch(mpatches.Rectangle((x1 - ox, y1 - oy), x2 - x1, y2 - y1,
                                        fill=False, edgecolor=PR_COLOR, linewidth=lw,
                                        linestyle=(0, (4, 2))))
        if show_labels:
            ax.text(x1 - ox, y2 - oy + 11, f"pred {CLASSES[cls]} {conf:.2f}", color=PR_COLOR,
                    fontsize=7.5, weight="bold",
                    bbox=dict(facecolor="black", alpha=0.55, pad=0.9, edgecolor="none"))


def collect(model, conf):
    """Run the detector over the test split and keep what each frame needs for plotting."""
    records = []
    files = sorted(TEST_IMG.glob("*.jpg"))
    for i in range(0, len(files), 32):
        batch = files[i:i + 32]
        for path, res in zip(batch, model.predict([str(p) for p in batch], conf=conf,
                                                  device="cpu", verbose=False)):
            h, w = res.orig_shape
            stem = path.name[:-4]
            gts = load_gt(stem, w, h)
            preds = [(int(b.cls), tuple(map(float, b.xyxy[0])), float(b.conf))
                     for b in res.boxes]
            label, best = verdict(gts, preds)
            records.append(dict(path=path, stem=stem, gts=gts, preds=preds,
                                verdict=label, iou=best,
                                cls=gts[0][0] if gts else None))
    return records


def pick(records):
    """One clear hit per class where possible, then the most instructive miss."""
    chosen, used = [], set()
    for cls in range(len(CLASSES)):
        hits = [r for r in records
                if r["cls"] == cls and r["verdict"] == "correct detection"]
        if hits:
            best = max(hits, key=lambda r: max(p[2] for p in r["preds"]))
            chosen.append(best)
            used.add(best["stem"])
    misses = [r for r in records if r["cls"] is not None
              and r["stem"] not in used and r["verdict"].startswith("MISSED")]
    if misses:
        chosen.append(misses[0])
    return chosen[:6]


def build(run, conf):
    from ultralytics import YOLO

    weights = HERE / "outputs" / "runs" / run / "weights" / "best.pt"
    model = YOLO(str(weights))
    records = collect(model, conf)

    hits = sum(r["verdict"] == "correct detection" for r in records)
    print(f"{hits}/{len(records)} test frames with a correct detection at conf {conf}")

    # Samples run across the page so the figure stays wide enough for a 16:9 slide:
    # top row is the full scene for context, bottom row the magnified target.
    samples = pick(records)
    cols = len(samples)
    fig, axes = plt.subplots(2, cols, figsize=(3.1 * cols, 7.6))
    axes = axes.reshape(2, cols)

    for c, rec in enumerate(samples):
        img = cv2.cvtColor(cv2.imread(str(rec["path"])), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes = [tuple(g[1:]) for g in rec["gts"]] + [p[1] for p in rec["preds"]]
        ok = rec["verdict"] == "correct detection"

        ax = axes[0][c]
        ax.imshow(img, cmap="gray")
        draw(ax, rec["gts"], rec["preds"], lw=1.4, show_labels=False)
        ax.set_title(f"{rec['stem'].split('_jpg')[0]}\nfull frame  {w}x{h}", fontsize=8)
        ax.axis("off")

        x1, y1, x2, y2 = crop_window(boxes, w, h)
        ax = axes[1][c]
        ax.imshow(img[y1:y2, x1:x2], cmap="gray")
        draw(ax, rec["gts"], rec["preds"], offset=(x1, y1), lw=2.0)
        note = f"IoU {rec['iou']:.2f}" if rec["iou"] > 0 else "no overlap"
        ax.set_title(f"zoom {x2 - x1}x{y2 - y1} px\n{rec['verdict']} ({note})",
                     fontsize=8.5, weight="bold",
                     color="#0B6B2E" if ok else "#B00020")
        ax.axis("off")

    fig.legend(handles=[
        mpatches.Patch(edgecolor=GT_COLOR, facecolor="none", linewidth=2,
                       label="Ground truth (annotation)"),
        mpatches.Patch(edgecolor=PR_COLOR, facecolor="none", linewidth=2,
                       label=f"Model prediction (conf \u2265 {conf})"),
    ], loc="upper center", ncol=2, fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, 0.945))
    fig.suptitle(f"{run}: ground truth vs prediction on held-out sequence-disjoint "
                 f"test frames", fontsize=12, y=0.985)
    # Set explicitly rather than via tight_layout, which collapses the gap the
    # two-line verdict titles on the bottom row need.
    fig.subplots_adjust(top=0.87, bottom=0.01, left=0.01, right=0.99,
                        hspace=0.22, wspace=0.06)

    out = FIG / "gt_vs_pred.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # The slide quotes these, so write them out instead of leaving them to be retyped.
    ious = [r["iou"] for r in samples if r["verdict"] == "correct detection"]
    (FIG / "gt_vs_pred_stats.json").write_text(json.dumps({
        "run": run,
        "conf": conf,
        "frames": len(records),
        "frames_with_correct_detection": hits,
        "shown": len(samples),
        "iou_min": round(min(ious), 2) if ious else None,
        "iou_max": round(max(ious), 2) if ious else None,
        "samples": [{"frame": r["stem"].split("_jpg")[0], "verdict": r["verdict"],
                     "iou": round(r["iou"], 3)} for r in samples],
    }, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="C_clean_yolov8s")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    print("wrote", build(args.run, args.conf))
