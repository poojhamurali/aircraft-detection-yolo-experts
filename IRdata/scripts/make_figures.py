"""Build the report figures.

Four figures, each answering one question a reader will ask:
  split_integrity   is the evaluation trustworthy?
  top3_comparison   which of the three finalist architectures is best?
  error_breakdown   what kind of mistakes does the best model make?
  predictions       what do those mistakes look like?
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent
FIG = HERE / "outputs" / "figures"
CLASSES = ["airplane", "bird", "drone", "helicopter"]
BEST_RUN = "top2_yolo11s"

# The three 25-epoch finalists, trained on the leakage-free `clean` split.
FINAL_RUNS = {
    "top1_yolo12s": "YOLO12s",
    "top2_yolo11s": "YOLO11s",
    "top3_yolov8s_p2": "YOLOv8s-P2",
}

# Earlier runs, kept only to show what split contamination does to the numbers.
# Every one of these was measured on a split that leaks, so none is a real result.
LEAKY_RUNS = {
    "A_official_yolov8n": ("as-shipped", 94.9),
    "B_clean_yolov8n": ("sequence-disjoint", 31.5),
    "baseline_yolo12s": ("sequence-disjoint", 31.5),
}


def _summary():
    return json.loads((HERE / "outputs" / "results_summary.json").read_text())


def split_integrity_chart():
    """Contamination per split variant, and what it is worth in mAP.

    The right panel deliberately shows only runs A and B. They share a model
    (YOLOv8n), an epoch budget (10) and an augmentation recipe, and differ in
    nothing but the split, so the gap between them is attributable to leakage
    alone. The clean split has no matched partner at that budget, so putting a
    third bar next to these two would invite a comparison the experiment does not
    support; the clean-split results live in their own figure instead.
    """
    summary = _summary()
    leak = [("as-shipped\nsplit", 94.9), ("sequence-disjoint\nsplit", 31.5),
            ("clean\nsplit", 0.0)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    bars = ax1.bar([n for n, _ in leak], [v for _, v in leak],
                   color=["#c0504d", "#e8a33d", "#4f9b57"])
    ax1.set_ylabel("% of val+test frames with a\nnear-identical twin in train")
    ax1.set_title("Evaluation-set contamination, by split")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3)
    for b, (_, v) in zip(bars, leak):
        ax1.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=10)

    pair = [("as-shipped\nsplit", "A_official_yolov8n"),
            ("sequence-disjoint\nsplit", "B_clean_yolov8n")]
    vals = [summary[r]["test"]["mAP50"] for _, r in pair]
    bars = ax2.bar([n for n, _ in pair], vals, color=["#c0504d", "#e8a33d"], width=0.55)
    ax2.set_ylabel("reported test mAP@0.5")
    ax2.set_title("Same model, same budget, only the split differs\n"
                  "(YOLOv8n, 10 epochs)", fontsize=11)
    ax2.set_ylim(0, 0.9)
    ax2.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center",
                 fontsize=11)
    ax2.annotate("", xy=(1, vals[1]), xytext=(1, vals[0]),
                 arrowprops=dict(arrowstyle="<->", color="#333"))
    ax2.text(1.06, (vals[0] + vals[1]) / 2, f"{vals[0] - vals[1]:+.3f} mAP\nbought by\nleakage",
             fontsize=9, va="center")

    fig.tight_layout()
    out = FIG / "split_integrity.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def top3_comparison_chart():
    """The three finalists, overall and per class, on the clean split."""
    summary = _summary()
    runs = [r for r in FINAL_RUNS if r in summary]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    metrics = ["mAP50", "mAP50_95", "precision", "recall"]
    width = 0.8 / len(runs)
    for i, run in enumerate(runs):
        vals = [summary[run]["test"][m] for m in metrics]
        b = ax1.bar([x + i * width for x in range(len(metrics))], vals, width,
                    label=FINAL_RUNS[run])
        for rect, v in zip(b, vals):
            ax1.text(rect.get_x() + rect.get_width() / 2, v + 0.008, f"{v:.3f}",
                     ha="center", fontsize=7)
    ax1.set_xticks([x + width for x in range(len(metrics))])
    ax1.set_xticklabels(["mAP@0.5", "mAP@0.5:0.95", "precision", "recall"])
    ax1.set_ylabel("test score")
    ax1.set_title("Overall test metrics (clean split, 25 epochs)")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    for i, run in enumerate(runs):
        pc = summary[run]["test"]["per_class"]
        vals = [pc.get(c, {}).get("mAP50", 0) for c in CLASSES]
        ax2.bar([x + i * width for x in range(len(CLASSES))], vals, width,
                label=FINAL_RUNS[run])
    ax2.set_xticks([x + width for x in range(len(CLASSES))])
    ax2.set_xticklabels(CLASSES)
    ax2.set_ylabel("test mAP@0.5")
    ax2.set_title("Per class: drone and helicopter solved, airplane and bird not")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG / "top3_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def error_breakdown_chart(run=BEST_RUN):
    """Confusion matrix plus the correct / wrong-class / missed split per class.

    These two panels together separate the two failure modes that a single mAP
    number fuses: calling an object the wrong name, and not seeing it at all.
    """
    path = HERE / "outputs" / "runs" / run / "confusion_test.json"
    if not path.exists():
        return None
    c = json.loads(path.read_text())
    labels = c["labels"]
    matrix = np.array(c["matrix_rows_truth_cols_pred"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    im = ax1.imshow(matrix, cmap="Blues")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("predicted")
    ax1.set_ylabel("ground truth")
    ax1.set_title(f"{FINAL_RUNS.get(run, run)}: confusion matrix (test, conf 0.25)")
    hi = matrix.max()
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax1.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=10,
                     color="white" if matrix[i, j] > hi * 0.5 else "black")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    pc = c["per_class"]
    correct = [pc[k]["correct_pct"] for k in CLASSES]
    wrong = [pc[k]["misclassified_pct"] for k in CLASSES]
    missed = [pc[k]["missed_pct"] for k in CLASSES]
    ax2.bar(CLASSES, correct, label="correct", color="#4f9b57")
    ax2.bar(CLASSES, wrong, bottom=correct, label="wrong class", color="#e8a33d")
    ax2.bar(CLASSES, missed, bottom=np.add(correct, wrong), label="missed entirely",
            color="#c0504d")
    for i, k in enumerate(CLASSES):
        if wrong[i] > 4:
            ax2.text(i, correct[i] + wrong[i] / 2, f"{wrong[i]:.0f}%", ha="center",
                     fontsize=9, color="black")
        ax2.text(i, correct[i] / 2, f"{correct[i]:.0f}%", ha="center", fontsize=9,
                 color="white")
        ax2.text(i, correct[i] + wrong[i] + missed[i] / 2, f"{missed[i]:.0f}%",
                 ha="center", fontsize=9, color="white")
    ax2.set_ylabel("% of ground-truth boxes")
    ax2.set_ylim(0, 100)
    ax2.set_title("Missed detections dominate, not class confusion")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG / "error_breakdown.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def prediction_montage(run=BEST_RUN, cols=3):
    src = HERE / "outputs" / "runs" / f"{run}_predictions"
    files = sorted(src.glob("*.jpg"))
    if not files:
        return None
    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for ax, path in zip(axes.ravel(), files):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(path.name.split("_jpg")[0], fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(files):]:
        ax.axis("off")
    fig.suptitle(f"{FINAL_RUNS.get(run, run)}: detections on held-out clean-split "
                 f"test frames (conf 0.25)", fontsize=12)
    fig.tight_layout()
    out = FIG / "predictions_montage.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for out in (split_integrity_chart(), top3_comparison_chart(),
                error_breakdown_chart(), prediction_montage()):
        if out:
            print("wrote", out)
