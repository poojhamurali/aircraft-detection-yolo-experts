"""Ground truth vs prediction for every IR model, on the same test frames.

One figure per training split, because a model may only be shown frames from its own
held-out test set: the 12-epoch screen and the leakage A/B were trained on
`seq_disjoint`, whose test frames are mostly training frames of `clean`, and vice
versa (only 20 frames, none with a bird or helicopter, are test frames in both).

Frames are picked by rule, not by eye: for each class, among test frames holding a
single box of that class, the one with the median box size. Each cell is a crop
centred on the ground-truth box (targets are ~26 px in a 640 px frame), with ground
truth in green and predictions (conf >= 0.25) in red. The cell border and caption give
the outcome at IoU >= 0.5: correct, wrong class, or missed; false alarms elsewhere in
the frame are counted underneath.

The model trained on the shipped `official` split is left out: its test frames have
byte-identical copies in its training set, so a gallery would show memorisation.

Usage:
    python scripts/make_model_gallery.py [--device cpu]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from confusion_analysis import iou_matrix, load_gt

HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "outputs" / "runs"
FIG = HERE / "outputs" / "figures"
CLASS_NAMES = ["airplane", "bird", "drone", "helicopter"]
CELL = 360
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

GROUPS = {
    "seq_disjoint": [
        ("bench_yolov8n", "YOLOv8n\n12 ep"),
        ("bench_yolov8n_p2", "YOLOv8n-P2\n12 ep"),
        ("bench_yolo11n", "YOLO11n\n12 ep"),
        ("bench_yolo12n", "YOLO12n\n12 ep"),
        ("bench_yolo26n", "YOLO26n\n12 ep"),
        ("bench_yolo26n_p2", "YOLO26n-P2\n12 ep"),
        ("B_clean_yolov8n", "YOLOv8n (A/B)\n10 ep"),
        ("C_clean_yolov8s", "YOLOv8s (A/B)\n10 ep"),
        ("baseline_yolo12s", "YOLO12s\n60 ep"),
    ],
    "clean": [
        ("top1_yolo12s", "YOLO12s\n25 ep"),
        ("top2_yolo11s", "YOLO11s\n25 ep"),
        ("top3_yolov8s_p2", "YOLOv8s-P2\n25 ep"),
        ("imp_v8s640_s1", "YOLOv8s\n15 ep"),
        ("imp_base640_s0", "YOLO11s\n15 ep"),
        ("imp_img960_s1", "YOLO11s 960 px\n15 ep"),
        ("imp_img1280_s0", "YOLO11s 1280 px\n15 ep"),
        ("imp_hard640_s1", "YOLO11s + rebal.\n15 ep"),
    ],
}
GREEN, RED, AMBER = (0, 200, 70), (235, 40, 40), (230, 160, 0)


def pick_frames(split):
    img_dir = HERE / "datasets" / split / "images" / "test"
    lbl_dir = HERE / "datasets" / split / "labels" / "test"
    chosen = []
    for c in range(len(CLASS_NAMES)):
        cands = []
        for img in sorted(img_dir.glob("*.jpg")):
            boxes, classes = load_gt(lbl_dir / f"{img.stem}.txt", 640, 640)
            if len(classes) == 1 and classes[0] == c:
                b = boxes[0]
                cands.append(((b[2] - b[0]) * (b[3] - b[1]), img, b))
        cands.sort(key=lambda t: (t[0], t[1].name))
        _, img, box = cands[len(cands) // 2]
        chosen.append({"cls": c, "image": img, "box": box})
    return chosen


def crop_window(box, w, h):
    bw, bh = box[2] - box[0], box[3] - box[1]
    side = int(min(max(5 * max(bw, bh), 128), min(w, h)))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0 = int(min(max(cx - side / 2, 0), w - side))
    y0 = int(min(max(cy - side / 2, 0), h - side))
    return x0, y0, side


def dashed_rect(d, xy, color, width=3, dash=10):
    x0, y0, x1, y1 = xy
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)),
                 ((x0, y1), (x0, y0))):
        length = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        for s in range(0, int(length), dash * 2):
            t0, t1 = s / max(length, 1), min(s + dash, length) / max(length, 1)
            d.line([(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                    (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)],
                   fill=color, width=width)


def render_cell(img, frame, preds=None):
    """Crop around the GT box, upscaled; returns (PIL image, outcome dict)."""
    w, h = img.size
    x0, y0, side = crop_window(frame["box"], w, h)
    scale = CELL / side
    cell = img.crop((x0, y0, x0 + side, y0 + side)).resize((CELL, CELL), Image.LANCZOS)
    cell = cell.convert("RGB")
    d = ImageDraw.Draw(cell)
    font = ImageFont.truetype(FONT_B, 20)

    def to_cell(b):
        return [(b[0] - x0) * scale, (b[1] - y0) * scale, (b[2] - x0) * scale,
                (b[3] - y0) * scale]

    d.rectangle(to_cell(frame["box"]), outline=GREEN, width=3)
    if preds is None:
        return cell, None

    boxes, confs, classes = preds
    outcome, matched = "missed", None
    ious = iou_matrix(boxes, [frame["box"]])[:, 0] if len(boxes) else np.zeros(0)
    if len(boxes):
        order = np.argsort(-confs)
        hits = [i for i in order if ious[i] >= 0.5]
        right = [i for i in hits if classes[i] == frame["cls"]]
        loose = [i for i in order if 0.1 <= ious[i] < 0.5 and classes[i] == frame["cls"]]
        if right:
            outcome, matched = "correct", right[0]
        elif hits:
            outcome, matched = f"wrong: {CLASS_NAMES[classes[hits[0]]]}", hits[0]
        elif loose:
            outcome, matched = f"loose box (IoU {ious[loose[0]]:.2f})", loose[0]
    # a false alarm is a box that does not touch the target at all
    false_alarms = int(sum(1 for i in range(len(boxes)) if i != matched and ious[i] < 0.1))
    for i, (b, cf, k) in enumerate(zip(boxes, confs, classes)):
        cb = to_cell(b)
        if cb[2] < 0 or cb[3] < 0 or cb[0] > CELL or cb[1] > CELL:
            continue
        dashed_rect(d, cb, RED)
        label = f"{CLASS_NAMES[k]} {cf:.2f}"
        ty = cb[1] - 24 if cb[1] > 26 else cb[3] + 4
        d.text((max(cb[0], 2), ty), label, fill=RED, font=font, stroke_width=2,
               stroke_fill=(0, 0, 0))
    return cell, {"outcome": outcome, "false_alarms": false_alarms}


def run_group(split, models, device):
    from ultralytics import YOLO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = pick_frames(split)
    images = [Image.open(f["image"]) for f in frames]
    results = {}
    cells = {}
    for run, _ in models:
        meta = json.loads((RUNS / run / "metrics.json").read_text())
        model = YOLO(str(RUNS / run / "weights" / "best.pt"))
        out = model.predict([str(f["image"]) for f in frames], imgsz=meta["imgsz"],
                            conf=0.25, iou=0.7, device=device, verbose=False)
        results[run] = {"test_mAP50": meta["test"]["mAP50"],
                        "test_mAP50_95": meta["test"]["mAP50_95"], "frames": []}
        for f, img, r in zip(frames, images, out):
            b = r.boxes
            cell, oc = render_cell(img, f, (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                            b.cls.cpu().numpy().astype(int)))
            cells[(run, f["cls"])] = (cell, oc)
            results[run]["frames"].append({"class": CLASS_NAMES[f["cls"]], **oc})

    cols = len(models) + 1
    fig, axes = plt.subplots(len(frames), cols, figsize=(1.55 * cols, 1.62 * len(frames)),
                             gridspec_kw=dict(wspace=0.04, hspace=0.3))
    border = {"correct": "#00A046", "missed": "#E02828"}
    for r, (f, img) in enumerate(zip(frames, images)):
        gt_cell, _ = render_cell(img, f)
        ax = axes[r, 0]
        ax.imshow(gt_cell)
        bw = f["box"][2] - f["box"][0]
        bh = f["box"][3] - f["box"][1]
        ax.set_xlabel(f"{CLASS_NAMES[f['cls']]}\n{bw:.0f}\u00d7{bh:.0f} px", fontsize=7.5,
                      labelpad=2, fontweight="bold")
        if r == 0:
            ax.set_title("Ground truth", fontsize=8, fontweight="bold")
        for c, (run, label) in enumerate(models, start=1):
            cell, oc = cells[(run, f["cls"])]
            ax = axes[r, c]
            ax.imshow(cell)
            colour = border.get(oc["outcome"], "#E6A000")
            for sp in ax.spines.values():
                sp.set_edgecolor(colour)
                sp.set_linewidth(2.5)
            txt = oc["outcome"].replace("loose box ", "loose ") + (
                f" +{oc['false_alarms']} FA" if oc["false_alarms"] else "")
            ax.set_xlabel(txt, fontsize=7.5, color=colour, labelpad=1.5)
            if r == 0:
                ax.set_title(f"{label}\nmAP50 {results[run]['test_mAP50']:.3f}", fontsize=7.5)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    out = FIG / f"model_gallery_{split}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    summary = {"split": split, "frames": [{"class": CLASS_NAMES[f["cls"]],
                                           "image": f["image"].name} for f in frames],
               "models": results}
    (FIG / f"model_gallery_{split}.json").write_text(json.dumps(summary, indent=2,
                                                                 default=float))
    print(f"{out}")
    for run, rr in results.items():
        print(f"  {run:18s} " + "  ".join(f"{x['class'][:5]}:{x['outcome']}"
                                         f"{'+' + str(x['false_alarms']) if x['false_alarms'] else ''}"
                                         for x in rr["frames"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    for split, models in GROUPS.items():
        run_group(split, models, args.device)


if __name__ == "__main__":
    main()
