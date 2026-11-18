"""Crisp SAR stream report (Objective 3), with IR–SAR comparison at the end.

Structure:
  1. Dataset description (+ sources)
  2. Sample images
  3. Preprocessing (duplicates / leakage removal)
  4. Model and results tables
  5. Training / analysis figures
  6. Conclusion
  7. Comparison with Infrared stream

    ../.venv/bin/python scripts/generate_report_docx.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
OUT = HERE / "outputs" / "reports" / "SAR_Report_Simple.docx"
FIG = HERE / "outputs" / "figures"
RUNS = HERE / "outputs" / "runs"
IR_FIG = ROOT / "IRdata" / "outputs" / "figures"
BEST = "clean_yolov8s"

audit = json.loads((HERE / "outputs" / "dataset_report.json").read_text())
metrics = json.loads((RUNS / BEST / "metrics.json").read_text())
S = audit["clean_summary"]
bt, bv = metrics["test"], metrics["val"]
srcs = audit["sources"]

# Infrared numbers for the comparison section
ir_audit = json.loads((ROOT / "IRdata" / "outputs" / "dataset_report.json").read_text())
ir_results = json.loads((ROOT / "IRdata" / "outputs" / "results_summary.json").read_text())
ir_bt = ir_results["top2_yolo11s"]["test"]
ir_pc = ir_bt["per_class"]
ir_cs = ir_audit["clean_split_summary"]

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


def figure(name, width=6.1, cap=None, base=FIG):
    path = (base / name) if not isinstance(name, Path) else name
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


def run_rows(name):
    with (RUNS / name / "results.csv").open() as fh:
        return list(csv.DictReader(fh))


def run_minutes(name):
    rows = run_rows(name)
    return float(rows[-1].get(" time") or rows[-1]["time"]) / 60


def best_epoch(name):
    rows = run_rows(name)
    def fitness(r):
        return (0.1 * float(r["metrics/mAP50(B)"]) +
                0.9 * float(r["metrics/mAP50-95(B)"]))
    b = max(rows, key=fitness)
    return int(float(b["epoch"])), float(b["metrics/mAP50(B)"])


# ---------------------------------------------------------------- title
title = doc.add_heading("SAR Aircraft Detection — SARDet-100K", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("Objective 3 · Synthetic Aperture Radar stream")
r.italic = True

# ---------------------------------------------------------------- 1
h1("1. Dataset Description")

p("Source: SARDet-100K (NeurIPS 2024), aircraft class only. Six-class multi-source "
  "SAR benchmark filtered to category `aircraft`. Patches are greyscale SAR "
  "amplitude (single channel replicated to 3 for YOLO).")

table(["Property", "Value"], [
    ["Source benchmark", "SARDet-100K"],
    ["Class retained", "aircraft (1 of 6)"],
    ["Images with aircraft", f"{audit['images_with_aircraft']}"],
    ["Aircraft boxes", f"{audit['aircraft_boxes']}"],
    ["Unique by content (MD5)", f"{audit['unique_images_by_content']}"],
    ["Median box side", f"{audit['median_box_side_px']:.1f} px"],
    ["Boxes < 32×32", f"{audit['pct_boxes_below_32x32']}%"],
    ["Boxes < 16×16", f"{audit['pct_boxes_below_16x16']}%"],
], widths=[2.6, 3.6], bold_first_col=True)

p("Constituent sources (identified by native frame size after SARDet renumbering):")

table(["Source", "Images", "Sensor", "Band / resolution", "Patch size"], [
    ["SAR-AIRcraft", srcs.get("SAR-AIRcraft", 0),
     "Gaofen-3", "C-band · 1 m", "512 × 512"],
    ["SADD", srcs.get("SADD", 0),
     "TerraSAR-X", "X-band · 0.5–3 m", "224 × 224"],
    ["MSAR", srcs.get("MSAR", 0),
     "HISEA-1", "C-band · <1 m", "256 × 256"],
], widths=[1.4, 0.9, 1.2, 1.5, 1.2], bold_first_col=True)

table(["Split", "Images", "Boxes", "SAR-AIRcraft", "SADD", "MSAR"], [
    ["Train", S["train"]["images"], S["train"]["boxes"],
     S["train"]["sources"].get("SAR-AIRcraft", 0),
     S["train"]["sources"].get("SADD", 0),
     S["train"]["sources"].get("MSAR", 0)],
    ["Val", S["val"]["images"], S["val"]["boxes"],
     S["val"]["sources"].get("SAR-AIRcraft", 0),
     S["val"]["sources"].get("SADD", 0),
     S["val"]["sources"].get("MSAR", 0)],
    ["Test", S["test"]["images"], S["test"]["boxes"],
     S["test"]["sources"].get("SAR-AIRcraft", 0),
     S["test"]["sources"].get("SADD", 0),
     S["test"]["sources"].get("MSAR", 0)],
], widths=[1.0, 0.9, 0.9, 1.2, 0.9, 0.9], bold_first_col=True)

# ---------------------------------------------------------------- 2
h1("2. Sample Images")

figure("dataset_samples.png", 6.2,
       "Figure 1. SAR aircraft patches with ground-truth boxes (green).")

figure("dataset_stats.png", 6.2,
       "Figure 2. Source composition, clean-split sizes, target-size histogram.")

# ---------------------------------------------------------------- 3
h1("3. Preprocessing")

p("SARDet-100K renumbers files and crops large scenes into patches. Exact "
  "duplicates and overlapping crops therefore leak across the published split.")

table(["Issue", "Count / rate"], [
    ["Exact duplicate files",
     f"{audit['duplicate_image_files']} / {audit['images_with_aircraft']}"],
    ["Unique images (MD5)", f"{audit['unique_images_by_content']}"],
    ["Duplicate groups crossing published splits",
     f"{audit['duplicate_groups_crossing_official_splits']}"],
    ["Official eval images with train near-twin",
     f"{audit['official_eval_images_with_train_twin']} / "
     f"{audit['official_eval_images']} "
     f"({audit['official_eval_leak_pct']}%)"],
    ["Near-dup threshold (64×64 RMSE)",
     f"{audit['visual_groups']['threshold']} "
     f"(unrelated-pair median RMSE "
     f"{audit['visual_groups']['unrelated_pair_rmse_median']})"],
], widths=[3.8, 2.4], bold_first_col=True)

p("Clean-split construction:")
bullet("Filter COCO annotations to `aircraft`; convert boxes to YOLO format.")
bullet("MD5 deduplication of byte-identical patches.")
bullet("Visual near-duplicate grouping (thumbnail RMSE; threshold calibrated so "
       "unrelated-pair false-positive rate ≈ 0.02%).")
bullet("Group-disjoint train/val/test so no near-identical group spans splits.")

figure("leakage_explained.png", 6.0,
       "Figure 3. Exact duplication and published-split contamination rates.")

# ---------------------------------------------------------------- 4
h1("4. Model and Results")

ep, _ = best_epoch(BEST)
mins = run_minutes(BEST)
epochs_done = metrics.get("epochs_completed", len(run_rows(BEST)))
epochs_planned = metrics.get("epochs_planned", 15)

p("Detector: YOLOv8s, fine-tuned from COCO-pretrained weights on the clean "
  "aircraft subset.")

table(["Setting", "Value"], [
    ["Model", "YOLOv8s (yolov8s.pt)"],
    ["Epochs completed", f"{epochs_done} of {epochs_planned}"],
    ["Image size", "512 (native for most patches)"],
    ["Batch", str(metrics.get("batch", 8))],
    ["Device", "Apple M5 Pro (MPS)"],
    ["Seed", "0"],
    ["Checkpoint kept", f"epoch {ep} (fitness = 0.1·mAP50 + 0.9·mAP50-95)"],
    ["Wall time", f"~{mins:.0f} min"],
    ["SAR-specific aug", "HSV off; rotation off (range/azimuth fixed); "
                         "flips 0.5; scale 0.40; mosaic on; mixup/copy-paste off"],
], widths=[2.4, 3.8], bold_first_col=True)

table(["Split", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"], [
    ["Val", f"{bv['mAP50']:.3f}", f"{bv['mAP50_95']:.3f}",
     f"{bv['precision']:.3f}", f"{bv['recall']:.3f}"],
    ["Test", f"{bt['mAP50']:.3f}", f"{bt['mAP50_95']:.3f}",
     f"{bt['precision']:.3f}", f"{bt['recall']:.3f}"],
], widths=[1.2, 1.2, 1.4, 1.2, 1.1], bold_first_col=True)

p(f"Note: training stopped at epoch {epochs_done}/{epochs_planned}; reported "
  f"scores are from the best of the completed rounds.")

# ---------------------------------------------------------------- 5
h1("5. Training Curves and Predictions")

figure("training_curve.png", 6.2,
       "Figure 4. Training losses and validation accuracy.")

figure("simple_results.png", 5.8,
       "Figure 5. Headline test metrics on the clean split.")

figure("predictions_montage.png", 5.8,
       "Figure 6. Predictions on held-out SAR patches.")

# ---------------------------------------------------------------- 6
h1("6. Conclusion")

p("Performance summary.")
bullet(f"Test mAP@0.5 = {bt['mAP50']:.3f}, mAP@0.5:0.95 = {bt['mAP50_95']:.3f}.")
bullet(f"Precision {bt['precision']:.3f}, recall {bt['recall']:.3f} "
       f"(~{(1 - bt['recall']) * 100:.0f}% of aircraft still missed).")
bullet(f"Strict mAP lags the headline ({bt['mAP50_95']:.3f} vs {bt['mAP50']:.3f}) "
       f"— typical for speckled radar localisation.")
bullet(f"Published split contamination: {audit['official_eval_leak_pct']}% "
       f"eval near-twins; {audit['duplicate_image_files']} exact duplicate files. "
       f"Clean split is the quoted result.")

p("Limits: single seed; incomplete epoch budget; official-vs-clean mAP gap not "
  "yet measured as a controlled A/B (contamination reported as a rate). "
  "Next: finish remaining epochs; per-source breakdown (SAR-AIRcraft / SADD / MSAR); "
  "controlled leakage A/B.")

# ---------------------------------------------------------------- 7 comparison
h1("7. Comparison: Infrared vs SAR")

p("Both streams use the same detection family (YOLO), leakage-aware splits, and "
  "MPS training. They differ in sensor, viewpoint, target scale, and difficulty.")

table(["Property", "Infrared (O2)", "SAR (O3)"], [
    ["Dataset", "IRdata-4465", "SARDet-100K (aircraft)"],
    ["Sensor", "Airborne thermal IR", "Spaceborne SAR (GF-3 / TerraSAR-X / HISEA-1)"],
    ["Classes", "airplane, bird, drone, helicopter", "aircraft"],
    ["Images (clean total)",
     str(ir_cs["train"]["images"] + ir_cs["val"]["images"] + ir_cs["test"]["images"]),
     str(S["train"]["images"] + S["val"]["images"] + S["test"]["images"])],
    ["Boxes (clean total)",
     str(ir_cs["train"]["boxes"] + ir_cs["val"]["boxes"] + ir_cs["test"]["boxes"]),
     str(S["train"]["boxes"] + S["val"]["boxes"] + S["test"]["boxes"])],
    ["Median target size",
     f"~{int(ir_audit['box_px']['median_area'] ** 0.5)} px",
     f"{audit['median_box_side_px']:.0f} px"],
    ["Share < 32×32",
     f"{ir_audit['box_px']['pct_below_32x32']}%",
     f"{audit['pct_boxes_below_32x32']}%"],
    ["Published leakage (eval)",
     f"{ir_audit['contaminated_val_test_files']} exact train copies",
     f"{audit['official_eval_leak_pct']}% near-twins"],
    ["Model quoted", "YOLO11s · 25 epochs",
     f"YOLOv8s · {epochs_done} epochs"],
    ["Test mAP@0.5", f"{ir_bt['mAP50']:.3f}", f"{bt['mAP50']:.3f}"],
    ["Test mAP@0.5:0.95", f"{ir_bt['mAP50_95']:.3f}", f"{bt['mAP50_95']:.3f}"],
    ["Precision", f"{ir_bt['precision']:.3f}", f"{bt['precision']:.3f}"],
    ["Recall", f"{ir_bt['recall']:.3f}", f"{bt['recall']:.3f}"],
], widths=[2.2, 2.3, 2.3], bold_first_col=True)

p("Infrared per-class detail (for contrast with single-class SAR):")
table(["IR class", "mAP@0.5", "Precision", "Recall"], [
    [c.capitalize(), f"{ir_pc[c]['mAP50']:.3f}",
     f"{ir_pc[c]['precision']:.3f}", f"{ir_pc[c]['recall']:.3f}"]
    for c in ("airplane", "bird", "drone", "helicopter")
] + [["IR overall", f"{ir_bt['mAP50']:.3f}",
      f"{ir_bt['precision']:.3f}", f"{ir_bt['recall']:.3f}"]],
     widths=[1.5, 1.3, 1.3, 1.3], bold_first_col=True)

figure("simple_results.png", 5.5,
       cap="Figure 7. Infrared per-class accuracy (YOLO11s, clean test).",
       base=IR_FIG)

figure("simple_results.png", 5.5,
       cap="Figure 8. SAR headline metrics (YOLOv8s, clean test).")

figure("dataset_samples.png", 5.8,
       cap="Figure 9. Infrared samples by class.",
       base=IR_FIG)

figure("dataset_samples.png", 5.8,
       cap="Figure 10. SAR aircraft samples.")

p("Reading the comparison.")
bullet(f"SAR test mAP@0.5 ({bt['mAP50']:.3f}) exceeds overall IR "
       f"({ir_bt['mAP50']:.3f}) and approaches IR drone performance "
       f"({ir_pc['drone']['mAP50']:.3f}).")
bullet("IR is multi-class and dominated by sub-32 px targets; SAR is single-class "
       "with larger median boxes — difficulty, not just model choice, drives the gap.")
bullet("Both streams required leakage cleanup; IR contamination was far more severe "
       "(exact train copies in eval) than SAR (4.6% near-twins).")
bullet("Complementarity for fusion (O4): IR covers airborne / night thermal; "
       "SAR covers all-weather nadir airport scenes. Decision-level fusion is the "
       "appropriate merge path (no co-registered triplets).")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"wrote {OUT}")
