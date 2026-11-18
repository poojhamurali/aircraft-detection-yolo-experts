"""Crisp Infrared stream report (Objective 2).

Structure:
  1. Dataset description (+ sources)
  2. Sample images
  3. Preprocessing (duplicates / leakage removal)
  4. Model and results tables
  5. Training / analysis figures
  6. Conclusion

    ../.venv/bin/python scripts/generate_simple_report_docx.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "outputs" / "reports" / "Infrared_Report_Simple.docx"
FIG = HERE / "outputs" / "figures"
RUNS = HERE / "outputs" / "runs"
CLASSES = ["airplane", "bird", "drone", "helicopter"]
BEST = "top2_yolo11s"
BEST_LABEL = "YOLO11s"

audit = json.loads((HERE / "outputs" / "dataset_report.json").read_text())
results = json.loads((HERE / "outputs" / "results_summary.json").read_text())
confusion = json.loads((RUNS / BEST / "confusion_test.json").read_text())

bt = results[BEST]["test"]
pc = bt["per_class"]
cf = confusion["per_class"]
cs = audit["clean_split_summary"]
gt_total = sum(d["gt_boxes"] for d in cf.values())
n_wrong = sum(d["misclassified"] for d in cf.values())
n_missed = sum(d["missed"] for d in cf.values())

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)


def h1(t):
    doc.add_heading(t, level=1)


def p(text, bold=False, italic=False):
    para = doc.add_paragraph()
    r = para.add_run(text)
    r.bold, r.italic = bold, italic
    return para


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def caption(text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = para.add_run(text)
    r.italic = True
    r.font.size = Pt(9)


def figure(name, width=6.1, cap=None):
    path = FIG / name
    if not path.exists():
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if cap:
        caption(cap)


def table(headers, rows, widths=None, bold_first_col=False):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, htxt in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(htxt))
        run.bold = True
        run.font.size = Pt(10)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(10)
            if bold_first_col and i == 0:
                run.bold = True
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Inches(w)
    doc.add_paragraph()
    return t


def run_minutes(name):
    path = RUNS / name / "results.csv"
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    return float(rows[-1].get(" time") or rows[-1]["time"]) / 60


def best_epoch(name):
    path = RUNS / name / "results.csv"
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    b = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    return int(float(b["epoch"])), float(b["metrics/mAP50(B)"])


# ---------------------------------------------------------------- title
title = doc.add_heading("Infrared Aircraft Detection — IRdata-4465", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("Objective 2 · Thermal / infrared stream")
r.italic = True

# ---------------------------------------------------------------- 1
h1("1. Dataset Description")

p("Source: IRdata-4465 — airborne thermal (infrared) imagery of airborne targets. "
  "Frames are single-channel heat maps rendered as 640×640 greyscale. No RGB colour "
  "information is present.")

p("Classes: airplane, bird, drone, helicopter.")

table(["Property", "Value"], [
    ["Source dataset", "IRdata-4465"],
    ["Sensor / modality", "Airborne thermal IR"],
    ["Frame size", "640 × 640 px"],
    ["Raw image files", f"{audit['images']}"],
    ["Unique by content (MD5)", f"{audit['unique_images_by_content']}"],
    ["Annotated boxes", f"{audit['boxes']}"],
    ["Sequences", f"{audit['sequences']}"],
    ["Median target side", f"~{int(audit['box_px']['median_area'] ** 0.5)} px"],
    ["Targets < 32×32", f"{audit['box_px']['pct_below_32x32']}%"],
    ["Targets < 16×16", f"{audit['box_px']['pct_below_16x16']}%"],
], widths=[2.4, 3.8], bold_first_col=True)

table(["Class", "Train", "Val", "Test", "Total (clean)"], [
    [c.capitalize(),
     cs["train"]["per_class"][c],
     cs["val"]["per_class"][c],
     cs["test"]["per_class"][c],
     cs["train"]["per_class"][c] + cs["val"]["per_class"][c] + cs["test"]["per_class"][c]]
    for c in CLASSES
] + [["All",
      cs["train"]["boxes"], cs["val"]["boxes"], cs["test"]["boxes"],
      cs["train"]["boxes"] + cs["val"]["boxes"] + cs["test"]["boxes"]]],
      widths=[1.3, 1.0, 1.0, 1.0, 1.4], bold_first_col=True)

# ---------------------------------------------------------------- 2
h1("2. Sample Images")

figure("dataset_samples.png", 6.2,
       "Figure 1. One sample per class with ground-truth boxes (green). "
       "Top: full frame. Bottom: magnified target.")

figure("dataset_stats.png", 6.2,
       "Figure 2. Class counts by split (left) and target-size distribution (right).")

# ---------------------------------------------------------------- 3
h1("3. Preprocessing")

p("The published split was contaminated. Exact and near-duplicate frames spanned "
  "training and evaluation, so scores from the as-shipped split are not trusted.")

table(["Issue", "Count / rate"], [
    ["Exact duplicate files (of raw set)",
     f"{audit['duplicate_image_files']} / {audit['images']}"],
    ["Unique images after MD5 dedup", f"{audit['unique_images_by_content']}"],
    ["Eval files that were exact train copies (published)",
     f"{audit['contaminated_val_test_files']}"],
    ["Sequences spanning published splits",
     f"{audit['sequences_spanning_official_splits']}"],
    ["Filename–label mismatches retained",
     f"{audit['filename_class_mismatch_count']}"],
], widths=[4.2, 2.0], bold_first_col=True)

p("Clean-split construction:")
bullet("MD5 deduplication of byte-identical frames.")
bullet("Union of sequence identity with visual near-duplicates "
       "(64×64 thumbnail RMSE; threshold calibrated against unrelated-pair baseline).")
bullet("Group-disjoint assignment so no near-identical group appears on both sides "
       "of the split.")
bullet("Independent verifier: 0 MD5 cross-split hits, 0 sequence overlap, "
       "0 near-duplicate leakage on the materialised clean set.")

p(f"Controlled leakage cost (YOLOv8n, 10 epochs, only split differs): "
  f"published {results['A_official_yolov8n']['test']['mAP50']:.3f} → "
  f"clean {results['B_clean_yolov8n']['test']['mAP50']:.3f} "
  f"(Δ ≈ {results['A_official_yolov8n']['test']['mAP50'] - results['B_clean_yolov8n']['test']['mAP50']:.3f} mAP@0.5).")

table(["Split", "Images", "Boxes", "Sequences", "Groups"], [
    ["Train", cs["train"]["images"], cs["train"]["boxes"],
     cs["train"]["sequences"], cs["train"]["groups"]],
    ["Val", cs["val"]["images"], cs["val"]["boxes"],
     cs["val"]["sequences"], cs["val"]["groups"]],
    ["Test", cs["test"]["images"], cs["test"]["boxes"],
     cs["test"]["sequences"], cs["test"]["groups"]],
], widths=[1.2, 1.1, 1.1, 1.2, 1.1], bold_first_col=True)

figure("split_integrity.png", 6.0,
       "Figure 3. Split integrity: controlled A/B leakage measurement.")

# ---------------------------------------------------------------- 4
h1("4. Model and Results")

p(f"Detector: Ultralytics YOLO, fine-tuned from COCO-pretrained weights. "
  f"Screened architectures → top-3 trained 25 epochs on the clean split. "
  f"Selected model: {BEST_LABEL}.")

ep, _ = best_epoch(BEST)
mins = run_minutes(BEST)

table(["Setting", "Value"], [
    ["Model", f"{BEST_LABEL} (yolo11s.pt)"],
    ["Epochs", "25"],
    ["Image size", "640"],
    ["Batch", "16"],
    ["Device", "Apple M5 Pro (MPS)"],
    ["Seed", "0"],
    ["Best epoch (by val mAP@0.5)", str(ep)],
    ["Wall time", f"~{mins:.0f} min"],
    ["IR-specific aug", "HSV off; rotation/flipud off; scale 0.30; "
                        "mosaic on (close last 3); mixup/copy-paste off"],
], widths=[2.4, 3.8], bold_first_col=True)

table(["Model", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"], [
    ["YOLO11s", f"{results['top2_yolo11s']['test']['mAP50']:.3f}",
     f"{results['top2_yolo11s']['test']['mAP50_95']:.3f}",
     f"{results['top2_yolo11s']['test']['precision']:.3f}",
     f"{results['top2_yolo11s']['test']['recall']:.3f}"],
    ["YOLOv8s-P2", f"{results['top3_yolov8s_p2']['test']['mAP50']:.3f}",
     f"{results['top3_yolov8s_p2']['test']['mAP50_95']:.3f}",
     f"{results['top3_yolov8s_p2']['test']['precision']:.3f}",
     f"{results['top3_yolov8s_p2']['test']['recall']:.3f}"],
    ["YOLO12s", f"{results['top1_yolo12s']['test']['mAP50']:.3f}",
     f"{results['top1_yolo12s']['test']['mAP50_95']:.3f}",
     f"{results['top1_yolo12s']['test']['precision']:.3f}",
     f"{results['top1_yolo12s']['test']['recall']:.3f}"],
], widths=[1.6, 1.1, 1.3, 1.1, 1.0], bold_first_col=True)

table(["Class", "mAP@0.5", "Precision", "Recall"], [
    [c.capitalize(), f"{pc[c]['mAP50']:.3f}",
     f"{pc[c]['precision']:.3f}", f"{pc[c]['recall']:.3f}"]
    for c in CLASSES
] + [["Overall", f"{bt['mAP50']:.3f}",
      f"{bt['precision']:.3f}", f"{bt['recall']:.3f}"]],
     widths=[1.5, 1.3, 1.3, 1.3], bold_first_col=True)

# ---------------------------------------------------------------- 5
h1("5. Training Curves and Error Analysis")

figure("training_curve.png", 6.2,
       "Figure 4. Training losses and validation accuracy over epochs.")

figure("simple_results.png", 5.8,
       "Figure 5. Per-class mAP@0.5 for YOLO11s on the clean test set.")

figure("top3_comparison.png", 6.0,
       "Figure 6. Top-3 model comparison.")

figure("error_breakdown.png", 6.2,
       "Figure 7. Confusion matrix and error breakdown "
       "(correct / wrong class / missed).")

table(["Class", "Correct %", "Wrong class %", "Missed %", "False positives"], [
    [c.capitalize(),
     f"{cf[c]['correct_pct']:.0f}%",
     f"{cf[c]['misclassified_pct']:.0f}%",
     f"{cf[c]['missed_pct']:.0f}%",
     cf[c]["false_positives"]]
    for c in CLASSES
], widths=[1.2, 1.1, 1.3, 1.1, 1.3], bold_first_col=True)

p(f"Of {gt_total} test objects: {n_wrong} wrong-class, {n_missed} missed "
  f"(~{n_missed / max(n_wrong, 1):.0f}:1). Misses dominate over class swaps. "
  f"Airplane→bird: {confusion['airplane_called_bird_pct']:.1f}%; "
  f"bird→airplane: {confusion['bird_called_airplane_pct']:.1f}%.")

figure("why_confused.png", 6.2,
       "Figure 8. Airplane vs bird at native pixel scale — classes collide at small sizes.")

figure("predictions_montage.png", 5.8,
       "Figure 9. Model predictions on held-out frames.")

# ---------------------------------------------------------------- 6
h1("6. Conclusion")

p("Performance summary.")
bullet(f"Overall test mAP@0.5 = {bt['mAP50']:.3f}, mAP@0.5:0.95 = {bt['mAP50_95']:.3f}.")
bullet(f"Drone ({pc['drone']['mAP50']:.3f}) and helicopter ({pc['helicopter']['mAP50']:.3f}) "
       f"are usable.")
bullet(f"Airplane ({pc['airplane']['mAP50']:.3f}) and bird ({pc['bird']['mAP50']:.3f}) "
       f"remain weak — small, low-contrast targets.")
bullet("Primary failure mode is missed detection, not class confusion.")
bullet("Published split leakage inflated scores by ~0.19 mAP; clean split is required.")

p("Limits: single seed; small per-class test counts; residual label noise "
  f"({audit['filename_class_mismatch_count']} filename conflicts). "
  "Next: temporal (video) cues, higher input resolution / tiling, longer P2 schedules.")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"wrote {OUT}")
