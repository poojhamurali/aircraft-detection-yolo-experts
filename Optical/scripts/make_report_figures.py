"""Figures for the plain-language optical report.

  dataset_samples.png   what a scene looks like, whole and zoomed to one aircraft
  dataset_stats.png     boxes per airport, and how big the targets are
  split_explained.png   why the shipped split could not be trusted
  leakage_comparison.png  both models scored on one common test set
  training_curve.png    loss and accuracy against epoch
  predictions_montage.png  the model's own output on unseen airports

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
DATA = HERE / "datasets" / "airport_disjoint"
RUNS = HERE / "outputs" / "runs"
BEST = "airport_disjoint_yolov8s"
GT = "#00E64D"

audit = json.loads((HERE / "outputs" / "dataset_report.json").read_text())
cross = json.loads((HERE / "outputs" / "cross_eval.json").read_text())


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
    """A whole scene with every aircraft boxed, then one aircraft magnified."""
    imgs = sorted((DATA / "images" / "test").glob("*.jpg"))
    # a scene with a useful number of aircraft: busy enough to be representative,
    # not so busy that the boxes merge into a solid block at page width
    pick = None
    for p in imgs:
        n = len(boxes_of(p.stem, "test", 1, 1))
        if 12 <= n <= 30:
            pick = p
            break
    pick = pick or imgs[0]

    img = cv2.cvtColor(cv2.imread(str(pick)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    bxs = boxes_of(pick.stem, "test", w, h)

    # The scene is 16:9 and the magnified crop is square, so equal-width panels
    # would render the scene as a short strip beside a tall box. Width ratios
    # matched to the aspect ratios make both panels the same height.
    fig = plt.figure(figsize=(13, 5.3))
    gs = fig.add_gridspec(1, 2, width_ratios=(w / h, 1.0), wspace=0.06,
                          top=0.80)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    for x1, y1, x2, y2 in bxs:
        ax1.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                         edgecolor=GT, linewidth=1.4))
    ax1.set_title(f"A whole scene ({pick.stem}) — {w} x {h} pixels\n"
                  f"{len(bxs)} labelled aircraft", fontsize=11, weight="bold")
    ax1.axis("off")

    # median-sized aircraft in this scene, not the biggest
    bxs_sorted = sorted(bxs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = bxs_sorted[len(bxs_sorted) // 2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * 2.2
    a, b = int(max(cx - half, 0)), int(max(cy - half, 0))
    c, d = int(min(cx + half, w)), int(min(cy + half, h))

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img[b:d, a:c])
    ax2.add_patch(mpatches.Rectangle((x1 - a, y1 - b), x2 - x1, y2 - y1, fill=False,
                                     edgecolor=GT, linewidth=2.4))
    ax2.set_title(f"One aircraft from that scene, magnified\n"
                  f"{int(x2 - x1)} x {int(y2 - y1)} pixels in the original image",
                  fontsize=11)
    ax2.axis("off")

    fig.suptitle("What the optical dataset looks like (green box = the human label)",
                 fontsize=13, y=0.97)
    out = FIG / "dataset_samples.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def dataset_stats():
    """How the airports divide up, and how big the targets are."""
    s = audit["airport_disjoint_split_summary"]

    sides = []
    for split in ("train", "val", "test"):
        for p in (DATA / "images" / split).glob("*.jpg"):
            for x1, y1, x2, y2 in boxes_of(p.stem, split, 4800, 2703):
                sides.append(((x2 - x1) * (y2 - y1)) ** 0.5)
    sides = np.array(sides)
    at_input = sides * (640.0 / 4800.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.5))

    names = ["train", "val", "test"]
    imgs = [s[n]["images"] for n in names]
    boxes = [s[n]["boxes"] for n in names]
    x = np.arange(3)
    ax1.bar(x - 0.2, imgs, 0.38, label="images", color="#4472c4")
    ax1.bar(x + 0.2, boxes, 0.38, label="aircraft", color="#ed7d31")
    for i, (a, b) in enumerate(zip(imgs, boxes)):
        ax1.text(i - 0.2, a + 60, str(a), ha="center", fontsize=9)
        ax1.text(i + 0.2, b + 60, str(b), ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n}\n{s[n]['airport_count']} airports" for n in names])
    ax1.set_title("How the data is divided\n(no airport appears in two groups)",
                  fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # A long tail of very large aircraft would otherwise fall outside the axis and
    # vanish without trace. Stack it into the last bar and label it.
    cap = 100
    n_over = int((at_input > cap).sum())
    ax2.hist(np.clip(at_input, 0, cap), bins=np.arange(0, cap + 3, 3),
             color="#4472c4", edgecolor="white")
    med = float(np.median(at_input))
    top = ax2.get_ylim()[1] * 1.2
    ax2.set_ylim(0, top)
    ax2.axvline(32, color="#c0504d", linestyle="--", linewidth=2)
    ax2.text(34, top * 0.96, '32 px: "small object"\nby the usual convention',
             fontsize=8.5, color="#c0504d", va="top")
    ax2.axvline(med, color="#2e7d32", linewidth=2)
    ax2.annotate(f"median {med:.0f} px", xy=(med, top * 0.55),
                 xytext=(med + 14, top * 0.72), fontsize=9, color="#2e7d32",
                 arrowprops=dict(arrowstyle="->", color="#2e7d32"))
    ax2.set_xlabel(f"aircraft size in pixels, as the model sees it after the\n"
                   f"4800 px scene is shrunk to the 640 px network input\n"
                   f"the {n_over} aircraft wider than {cap} px are stacked into the "
                   f"last bar", fontsize=9)
    ax2.set_ylabel("number of aircraft")
    ax2.set_title(f"How big the targets are: {100 * (at_input < 32).mean():.0f}% "
                  f"under 32 px at the input", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG / "dataset_stats.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def split_explained():
    """The two defects in the shipped split, as plain counts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.3))

    ax1.barh(["validation\nimages", "test\nimages"],
             [audit["shipped_val_images"], audit["shipped_test_images"]],
             color=["#c0504d", "#4472c4"], height=0.5)
    ax1.text(8, 0, f"{audit['shipped_val_images']} — the folder is empty;\n"
                   f"data.yaml pointed validation at the test set",
             va="center", fontsize=10, color="#c0504d", weight="bold")
    ax1.text(audit["shipped_test_images"] * 0.5, 1,
             f"{audit['shipped_test_images']}", va="center", ha="center",
             fontsize=11, color="white", weight="bold")
    ax1.set_xlim(0, audit["shipped_test_images"] * 1.15)
    ax1.set_title("Defect 1: there was no validation set\n"
                  "the model was tuned on the images it was scored on", fontsize=11)
    ax1.grid(axis="x", alpha=0.3)

    pct = audit["shipped_test_airport_overlap_pct"]
    ax2.barh(["shipped\ntest set"], [pct], color="#c0504d", height=0.38)
    ax2.barh(["our\ntest set"], [0], color="#4f9b57", height=0.38)
    ax2.text(pct - 3, 0, f"{pct:.0f}%", va="center", ha="right", fontsize=12,
             color="white", weight="bold")
    ax2.text(2, 1, "0% — four airports held out entirely", va="center", fontsize=10,
             color="#2e7d32", weight="bold")
    ax2.set_xlim(0, 110)
    ax2.set_xlabel("share of test images whose airport also appears in training")
    ax2.set_title("Defect 2: every test airport was also a training airport",
                  fontsize=11)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Why the published score could not be trusted", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG / "split_explained.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def leakage_comparison():
    """Both models on one common test set: the controlled measurement."""
    a = cross["random_split_yolov8s"]
    b = cross["airport_disjoint_yolov8s"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.6))

    labels = ["mAP@0.5\n(normal scoring)", "mAP@0.5:0.95\n(strict scoring)"]
    x = np.arange(2)
    va = [a["mAP50"], a["mAP50_95"]]
    vb = [b["mAP50"], b["mAP50_95"]]
    r1 = ax1.bar(x - 0.2, va, 0.38, color="#c0504d",
                 label="model that trained on 68% of these images")
    r2 = ax1.bar(x + 0.2, vb, 0.38, color="#4f9b57",
                 label="model that never saw these airports")
    for rects, vals in ((r1, va), (r2, vb)):
        for rect, v in zip(rects, vals):
            ax1.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.3f}",
                     ha="center", fontsize=10, weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("accuracy")
    ax1.set_title("Both models scored on the SAME 317 unseen-airport images",
                  fontsize=11)
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    gaps = [a["mAP50"] - b["mAP50"], a["mAP50_95"] - b["mAP50_95"]]
    cols = ["#c0504d" if g > 0 else "#4f9b57" for g in gaps]
    bars = ax2.bar(labels, gaps, color=cols, width=0.5)
    for bar, g in zip(bars, gaps):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 g + (0.0015 if g > 0 else -0.004), f"{g:+.3f}",
                 ha="center", va="bottom" if g > 0 else "top", fontsize=12,
                 weight="bold")
    ax2.axhline(0, color="#333", linewidth=1)
    ax2.set_ylim(min(gaps) - 0.012, max(gaps) + 0.014)
    ax2.set_ylabel("advantage from having seen the test images")
    ax2.set_title("What the cheating is worth\n"
                  "almost nothing, and nothing at all under strict scoring",
                  fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG / "leakage_comparison.png"
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
    ax1.set_xlabel("epoch (one pass over the training images)")
    ax1.set_ylabel("error")
    ax1.set_title("Training error falls steadily")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    m50 = [float(r["metrics/mAP50(B)"]) for r in rows]
    m95 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    ax2.plot(ep, m50, linewidth=2, color="#2e7d32", label="mAP@0.5")
    ax2.plot(ep, m95, linewidth=2, color="#4472c4", label="mAP@0.5:0.95 (stricter)")

    # Mark the epoch actually kept, not the epoch with the highest mAP@0.5.
    # Ultralytics selects best.pt by fitness = 0.1*mAP50 + 0.9*mAP50-95, which is
    # dominated by the strict metric, so the two rarely coincide: here mAP@0.5
    # peaks late while the saved checkpoint comes from much earlier. Annotating
    # the mAP@0.5 peak would advertise a score no saved model achieves.
    fitness = [0.1 * a + 0.9 * b for a, b in zip(m50, m95)]
    peak = int(np.argmax(fitness))
    ax2.scatter([ep[peak]], [m50[peak]], s=70, color="#2e7d32", zorder=5)
    ax2.annotate(f"checkpoint kept: epoch {int(ep[peak])}\n"
                 f"({m50[peak]:.3f} / {m95[peak]:.3f})",
                 xy=(ep[peak], m50[peak]), xytext=(6, -46),
                 textcoords="offset points", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#2e7d32"))
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Accuracy on held-out airports rises and then flattens")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = FIG / "training_curve.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def predictions_montage(n=4):
    src = RUNS / f"{BEST}_predictions"
    files = sorted(src.glob("*.jpg"))[:n]
    if not files:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.4))
    for ax, f in zip(axes.ravel(), files):
        ax.imshow(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB))
        ax.set_title(f.stem, fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[len(files):]:
        ax.axis("off")
    fig.suptitle("The model's own output on airports it was never trained on\n"
                 "every box is a detection the model made, with its confidence",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG / "predictions_montage.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for out in (dataset_samples(), dataset_stats(), split_explained(),
                leakage_comparison(), training_curve(), predictions_montage()):
        if out:
            print("wrote", out)
