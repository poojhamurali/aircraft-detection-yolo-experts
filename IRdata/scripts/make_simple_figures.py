"""Figures for the plain-language report.

The technical report's figures assume the reader already knows what an mAP is and
what the imagery looks like. These are aimed at someone seeing the project for the
first time, so they show the data itself before they show any metric.

  dataset_samples.png   what the infrared frames actually look like, per class
  why_confused.png      airplane and bird crops at true pixel scale, side by side
  dataset_stats.png     how many boxes per class, and how small they are
  simple_results.png    one bar chart of per-class accuracy
  training_curve.png    loss and accuracy against epoch

Usage:
    python scripts/make_simple_figures.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent
FIG = HERE / "outputs" / "figures"
DATA = HERE / "datasets" / "clean"
CLASSES = ["airplane", "bird", "drone", "helicopter"]
BEST = "top2_yolo11s"
GT_COLOR = "#00E64D"


def boxes_of(stem: str, split: str, w: int, h: int):
    """YOLO label file -> [(class, x1, y1, x2, y2)] in pixels."""
    path = DATA / "labels" / split / f"{stem}.txt"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
        out.append((cls, (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def index_by_class(split="train"):
    """One list of (image_path, box, area) per class, smallest target first."""
    per_class = {c: [] for c in range(len(CLASSES))}
    for img in sorted((DATA / "images" / split).glob("*.jpg")):
        for cls, x1, y1, x2, y2 in boxes_of(img.stem, split, 640, 640):
            per_class[cls].append((img, (x1, y1, x2, y2), (x2 - x1) * (y2 - y1)))
    for c in per_class:
        per_class[c].sort(key=lambda t: t[2])
    return per_class


def representative(entries, quantile=0.6):
    """The entry at a given size quantile for its class.

    Picking the largest target would misrepresent the dataset: a handful of
    airplane boxes span most of the frame while the median airplane is ~24 px, so
    the biggest example makes the task look far easier than it is. A slightly
    above-median target is typical and still legible once magnified.
    """
    if not entries:
        return None
    return entries[min(int(len(entries) * quantile), len(entries) - 1)]


def crop_window(box, w, h, zoom=4.0, min_size=120):
    """Square window centred on a box, padded by `zoom`, clipped to the frame."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    size = max(max(x2 - x1, y2 - y1) * zoom, min_size)
    half = min(size / 2, w / 2, h / 2)
    cx = min(max(cx, half), w - half)
    cy = min(max(cy, half), h - half)
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def dataset_samples():
    """Full frame above, magnified target below, one column per class."""
    per_class = index_by_class("train")
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))

    for col, cls in enumerate(range(len(CLASSES))):
        entry = representative(per_class[cls])
        if entry is None:
            continue
        img_path, box, _ = entry
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box

        ax = axes[0][col]
        ax.imshow(img)
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                        edgecolor=GT_COLOR, linewidth=1.6))
        ax.set_title(f"{CLASSES[cls].upper()}\nwhole frame, 640 x 640 px",
                     fontsize=10.5, weight="bold")
        ax.axis("off")

        cx1, cy1, cx2, cy2 = crop_window(box, w, h, zoom=5.0)
        ax = axes[1][col]
        ax.imshow(img[cy1:cy2, cx1:cx2])
        ax.add_patch(mpatches.Rectangle((x1 - cx1, y1 - cy1), x2 - x1, y2 - y1,
                                        fill=False, edgecolor=GT_COLOR, linewidth=2.2))
        ax.set_title(f"same target, zoomed in\nit is only "
                     f"{int(x2 - x1)} x {int(y2 - y1)} pixels", fontsize=9.5)
        ax.axis("off")

    fig.suptitle("What the dataset looks like: infrared frames with their labels "
                 "(green box = the annotation)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.0)
    out = FIG / "dataset_samples.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def why_confused(n=6):
    """Airplane and bird crops at identical scale, to show the classes really do look alike."""
    per_class = index_by_class("train")
    rng = random.Random(0)

    def sample(cls):
        # mid-sized targets: these are the ones the model actually confuses
        entries = per_class[cls]
        mid = entries[len(entries) // 3: 2 * len(entries) // 3] or entries
        return rng.sample(mid, min(n, len(mid)))

    # entries are sorted smallest-first by index_by_class

    fig, axes = plt.subplots(2, n, figsize=(2.05 * n, 4.9))
    for row, cls in enumerate((CLASSES.index("airplane"), CLASSES.index("bird"))):
        for col, (img_path, box, _) in enumerate(sample(cls)):
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            cx1, cy1, cx2, cy2 = crop_window(box, w, h, zoom=3.0, min_size=64)
            ax = axes[row][col]
            ax.imshow(img[cy1:cy2, cx1:cx2])
            ax.set_title(f"{int(box[2] - box[0])} px wide", fontsize=8)
            ax.axis("off")
        axes[row][0].set_ylabel(CLASSES[cls], fontsize=12)
        # ylabel is invisible with axis off, so annotate outside the image instead
        axes[row][0].text(-0.12, 0.5, CLASSES[cls].upper(), transform=axes[row][0].transAxes,
                          rotation=90, va="center", ha="center", fontsize=12, weight="bold")

    fig.suptitle("Why airplane and bird get mixed up: at these sizes the two classes "
                 "carry almost the same information", fontsize=12.5)
    fig.tight_layout(rect=(0.02, 0, 1, 0.93))
    out = FIG / "why_confused.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def dataset_stats():
    """Boxes per class, and the distribution of target sizes."""
    audit = json.loads((HERE / "outputs" / "dataset_report.json").read_text())
    cs = audit["clean_split_summary"]

    sides = []
    for split in ("train", "val", "test"):
        for img in (DATA / "images" / split).glob("*.jpg"):
            for _, x1, y1, x2, y2 in boxes_of(img.stem, split, 640, 640):
                sides.append(((x2 - x1) * (y2 - y1)) ** 0.5)
    sides = np.array(sides)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    width = 0.26
    for i, split in enumerate(("train", "val", "test")):
        vals = [cs[split]["per_class"][c] for c in CLASSES]
        b = ax1.bar([x + i * width for x in range(len(CLASSES))], vals, width,
                    label=f"{split} ({cs[split]['images']} images)")
        for rect, v in zip(b, vals):
            ax1.text(rect.get_x() + rect.get_width() / 2, v + 8, str(v), ha="center",
                     fontsize=8)
    ax1.set_xticks([x + width for x in range(len(CLASSES))])
    ax1.set_xticklabels(CLASSES)
    ax1.set_ylabel("number of labelled objects")
    ax1.set_title("How many examples of each class")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # A handful of boxes span most of the frame, which flattens everything of
    # interest into the first two bins. Clip the axis and say so, rather than
    # rescaling and letting the reader assume the tail does not exist.
    cap = 120
    n_over = int((sides > cap).sum())
    ax2.hist(np.clip(sides, 0, cap), bins=np.arange(0, cap + 4, 4),
             color="#4472c4", edgecolor="white")
    med = float(np.median(sides))
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.26)  # headroom so the labels clear the bars
    top = ax2.get_ylim()[1]
    ax2.axvline(32, color="#c0504d", linestyle="--", linewidth=2)
    ax2.text(36, top * 0.97, '32 px: to the left of this line an object\n'
                             'counts as "small" by the usual convention',
             fontsize=8.5, color="#c0504d", va="top")
    ax2.axvline(med, color="#2e7d32", linewidth=2)
    ax2.annotate(f"median {med:.0f} px", xy=(med, top * 0.78),
                 xytext=(med - 30, top * 0.9), fontsize=9, color="#2e7d32",
                 arrowprops=dict(arrowstyle="->", color="#2e7d32"))
    ax2.set_xlabel(f"target size in pixels across (frame is 640 px wide)\n"
                   f"the {n_over} boxes wider than {cap} px are stacked into the last bar",
                   fontsize=9)
    ax2.set_ylabel("number of objects")
    ax2.set_title(f"How big the targets are: "
                  f"{100 * (sides < 32).mean():.0f}% are under 32 px")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG / "dataset_stats.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def simple_results():
    """One chart: how well the chosen model does on each class, plus the overall score."""
    summary = json.loads((HERE / "outputs" / "results_summary.json").read_text())
    t = summary[BEST]["test"]
    pc = t["per_class"]

    labels = [c.capitalize() for c in CLASSES] + ["ALL\nclasses"]
    vals = [pc[c]["mAP50"] for c in CLASSES] + [t["mAP50"]]
    colors = ["#c0504d" if v < 0.5 else "#4f9b57" for v in vals[:-1]] + ["#4472c4"]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                fontsize=11, weight="bold")
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=1.4)
    ax.text(-0.42, 0.515, "half marks", fontsize=8.5, color="#666")
    ax.set_ylabel("accuracy (mAP@0.5)")
    ax.set_ylim(0, 1.05)
    ax.set_title("YOLO11s accuracy by class, on images it was never trained on\n"
                 "green = works, red = does not work yet", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG / "simple_results.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def training_curve(run=BEST):
    """Loss going down and accuracy going up, which is what 'training worked' looks like."""
    path = HERE / "outputs" / "runs" / run / "results.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    ep = [float(r["epoch"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
    for key, lab in (("train/box_loss", "box loss (where the object is)"),
                     ("train/cls_loss", "class loss (what the object is)")):
        ax1.plot(ep, [float(r[key]) for r in rows], linewidth=2, label=lab)
    ax1.set_xlabel("epoch (one pass over the training images)")
    ax1.set_ylabel("error")
    ax1.set_title("Training error falls steadily")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    m50 = [float(r["metrics/mAP50(B)"]) for r in rows]
    ax2.plot(ep, m50, linewidth=2, color="#2e7d32", label="mAP@0.5")
    ax2.plot(ep, [float(r["metrics/mAP50-95(B)"]) for r in rows], linewidth=2,
             color="#4472c4", label="mAP@0.5:0.95 (stricter)")
    peak = int(np.argmax(m50))
    ax2.scatter([ep[peak]], [m50[peak]], s=70, color="#2e7d32", zorder=5)
    ax2.annotate(f"best: {m50[peak]:.3f} at epoch {int(ep[peak])}",
                 xy=(ep[peak], m50[peak]), xytext=(-10, -28),
                 textcoords="offset points", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#2e7d32"))
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Validation accuracy rises and then flattens")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = FIG / "training_curve.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for out in (dataset_samples(), why_confused(), dataset_stats(),
                simple_results(), training_curve()):
        if out:
            print("wrote", out)
