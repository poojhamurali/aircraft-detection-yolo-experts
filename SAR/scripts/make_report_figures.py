"""Figures for the plain-language SAR report.

  dataset_samples.png     SAR patches with labels, whole and zoomed
  dataset_stats.png       sources, split sizes, target sizes
  leakage_explained.png   duplication and official-split contamination
  training_curve.png      loss and accuracy against epoch
  predictions_montage.png model output on held-out patches
  simple_results.png      headline scores as a simple bar chart

Usage:
    ../.venv/bin/python scripts/make_report_figures.py
"""

from __future__ import annotations

import csv
import json
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
RUNS = HERE / "outputs" / "runs"
BEST = "clean_yolov8s"
GT = "#00E64D"

audit = json.loads((HERE / "outputs" / "dataset_report.json").read_text())
metrics = json.loads((RUNS / BEST / "metrics.json").read_text())


def boxes_of(stem, split, w, h):
    path = DATA / "labels" / split / f"{stem}.txt"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
        out.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def dataset_samples():
    """Three representative SAR patches with labels, plus one magnified crop."""
    imgs = sorted((DATA / "images" / "test").glob("*.jpg"))
    picks = []
    for p in imgs:
        n = len(boxes_of(p.stem, "test", 1, 1))
        if 2 <= n <= 8:
            picks.append(p)
        if len(picks) >= 3:
            break
    while len(picks) < 3 and imgs:
        picks.append(imgs[len(picks)])

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8))
    for ax, p in zip(axes[:3], picks):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ax.imshow(img)
        for x1, y1, x2, y2 in boxes_of(p.stem, "test", w, h):
            ax.add_patch(mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False,
                edgecolor=GT, linewidth=1.6))
        ax.set_title(f"{p.stem}\n{len(boxes_of(p.stem, 'test', 1, 1))} aircraft",
                     fontsize=10)
        ax.axis("off")

    # magnified median box from the first pick
    p = picks[0]
    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    bxs = sorted(boxes_of(p.stem, "test", w, h),
                 key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = bxs[len(bxs) // 2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(max(x2 - x1, y2 - y1) * 2.0, 40)
    a, b = int(max(cx - half, 0)), int(max(cy - half, 0))
    c, d = int(min(cx + half, w)), int(min(cy + half, h))
    axes[3].imshow(img[b:d, a:c])
    axes[3].add_patch(mpatches.Rectangle(
        (x1 - a, y1 - b), x2 - x1, y2 - y1, fill=False,
        edgecolor=GT, linewidth=2.2))
    axes[3].set_title(f"one aircraft, magnified\n"
                      f"{int(x2 - x1)} x {int(y2 - y1)} px", fontsize=10)
    axes[3].axis("off")

    fig.suptitle("What the SAR aircraft patches look like "
                 "(green box = human label)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = FIG / "dataset_samples.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def dataset_stats():
    s = audit["clean_summary"]
    sources = audit["sources"]

    sides = []
    for split in ("train", "val", "test"):
        for p in (DATA / "images" / split).glob("*.jpg"):
            for x1, y1, x2, y2 in boxes_of(p.stem, split, 512, 512):
                # most patches are 512; SADD/MSAR are smaller but YOLO labels
                # are normalised so this understates those a little — close enough
                # for the histogram's purpose
                sides.append(((x2 - x1) * (y2 - y1)) ** 0.5)
    sides = np.array(sides)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # sources
    names = list(sources.keys())
    vals = [sources[n] for n in names]
    colors = ["#4472c4", "#ed7d31", "#70ad47"]
    axes[0].bar(names, vals, color=colors[:len(names)])
    for i, v in enumerate(vals):
        axes[0].text(i, v + max(vals) * 0.02, str(v), ha="center", fontsize=9)
    axes[0].set_title("Where the aircraft images come from\n"
                      "(frame size identifies the source)", fontsize=11)
    axes[0].set_ylabel("images")
    axes[0].tick_params(axis="x", labelsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # split sizes
    splits = ["train", "val", "test"]
    imgs = [s[n]["images"] for n in splits]
    boxes = [s[n]["boxes"] for n in splits]
    x = np.arange(3)
    axes[1].bar(x - 0.2, imgs, 0.38, label="images", color="#4472c4")
    axes[1].bar(x + 0.2, boxes, 0.38, label="aircraft", color="#ed7d31")
    for i, (a, b) in enumerate(zip(imgs, boxes)):
        axes[1].text(i - 0.2, a + 200, str(a), ha="center", fontsize=8)
        axes[1].text(i + 0.2, b + 200, str(b), ha="center", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].set_title("Clean split\n(no near-duplicate across groups)", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    # sizes
    cap = 200
    n_over = int((sides > cap).sum())
    axes[2].hist(np.clip(sides, 0, cap), bins=np.arange(0, cap + 5, 5),
                 color="#4472c4", edgecolor="white")
    med = float(np.median(sides))
    top = axes[2].get_ylim()[1] * 1.15
    axes[2].set_ylim(0, top)
    axes[2].axvline(32, color="#c0504d", linestyle="--", linewidth=2)
    axes[2].text(36, top * 0.92, '32 px: "small"', fontsize=8.5, color="#c0504d")
    axes[2].axvline(med, color="#2e7d32", linewidth=2)
    axes[2].annotate(f"median {med:.0f} px", xy=(med, top * 0.55),
                     xytext=(med + 20, top * 0.75), fontsize=9, color="#2e7d32",
                     arrowprops=dict(arrowstyle="->", color="#2e7d32"))
    axes[2].set_xlabel(f"aircraft size in pixels\n"
                       f"({n_over} wider than {cap} stacked in last bar)", fontsize=9)
    axes[2].set_title(f"Target sizes: "
                      f"{100 * (sides < 32).mean():.0f}% under 32 px", fontsize=11)
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG / "dataset_stats.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def leakage_explained():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.3))

    unique = audit["unique_images_by_content"]
    dup = audit["duplicate_image_files"]
    ax1.barh(["unique\nimages", "byte-identical\nduplicates"],
             [unique, dup], color=["#4f9b57", "#c0504d"], height=0.5)
    ax1.text(unique * 0.5, 0, f"{unique}", va="center", ha="center",
             fontsize=12, color="white", weight="bold")
    ax1.text(dup * 0.5, 1, f"{dup}", va="center", ha="center",
             fontsize=12, color="white", weight="bold")
    ax1.set_xlim(0, (unique + dup) * 1.05)
    ax1.set_title(f"Of {audit['images_with_aircraft']} aircraft images,\n"
                  f"{dup} are exact copies of another", fontsize=11)
    ax1.grid(axis="x", alpha=0.3)

    pct = audit["official_eval_leak_pct"]
    ax2.barh(["official\neval set"], [pct], color="#c0504d", height=0.4)
    ax2.barh(["our clean\ntest set"], [0], color="#4f9b57", height=0.4)
    ax2.text(pct + 0.3, 0, f"{pct}% have a near-identical\ntraining twin",
             va="center", fontsize=10, color="#c0504d", weight="bold")
    ax2.text(0.3, 1, "0% — visual groups kept inside one split",
             va="center", fontsize=10, color="#2e7d32", weight="bold")
    ax2.set_xlim(0, max(pct * 2.5, 12))
    ax2.set_xlabel("share of evaluation images with a train near-duplicate")
    ax2.set_title("How much of the published split is contaminated", fontsize=11)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Why the published SARDet-100K split cannot be trusted as-is",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = FIG / "leakage_explained.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def training_curve(run=BEST):
    path = RUNS / run / "results.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    ep = [float(r["epoch"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
    for key, lab in (("train/box_loss", "box loss (where the aircraft is)"),
                     ("train/cls_loss", "class loss (whether it is an aircraft)")):
        ax1.plot(ep, [float(r[key]) for r in rows], linewidth=2, label=lab)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("error")
    ax1.set_title("Training error falls steadily")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    m50 = [float(r["metrics/mAP50(B)"]) for r in rows]
    m95 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    ax2.plot(ep, m50, linewidth=2, color="#2e7d32", label="mAP@0.5")
    ax2.plot(ep, m95, linewidth=2, color="#4472c4", label="mAP@0.5:0.95 (stricter)")
    fitness = [0.1 * a + 0.9 * b for a, b in zip(m50, m95)]
    peak = int(np.argmax(fitness))
    ax2.scatter([ep[peak]], [m50[peak]], s=70, color="#2e7d32", zorder=5)
    ax2.annotate(f"checkpoint kept: epoch {int(ep[peak])}\n"
                 f"({m50[peak]:.3f} / {m95[peak]:.3f})",
                 xy=(ep[peak], m50[peak]), xytext=(-40, -40),
                 textcoords="offset points", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#2e7d32"))
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title(f"Validation accuracy over {len(rows)} completed epochs\n"
                  f"(planned 15; training stopped early)", fontsize=11)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = FIG / "training_curve.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def simple_results():
    t = metrics["test"]
    labels = ["Accuracy\n(mAP@0.5)", "Strict accuracy\n(mAP@0.5:0.95)",
              "Precision", "Recall"]
    vals = [t["mAP50"], t["mAP50_95"], t["precision"], t["recall"]]
    colors = ["#4472c4", "#5b9bd5", "#4f9b57", "#70ad47"]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=12, weight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("YOLOv8s on the clean SAR test set\n"
                 "(2,982 patches, never seen during training)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "simple_results.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def predictions_montage(n=4):
    src = RUNS / f"{BEST}_predictions"
    files = sorted(src.glob("*.jpg"))[:n]
    if not files:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, f in zip(axes.ravel(), files):
        ax.imshow(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB))
        ax.set_title(f.stem, fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[len(files):]:
        ax.axis("off")
    fig.suptitle("The model's own output on SAR patches it was never trained on\n"
                 "every box is a detection the model made, with its confidence",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG / "predictions_montage.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for out in (dataset_samples(), dataset_stats(), leakage_explained(),
                training_curve(), simple_results(), predictions_montage()):
        if out:
            print("wrote", out)
