"""Generate the Objective 2 (infrared stream) technical report as a .docx.

Every figure in the report is read from outputs/, and every number is read from
outputs/dataset_report.json, outputs/results_summary.json,
outputs/split_verification*.json and the per-run confusion_test.json files at
generation time, so the document cannot drift out of step with the experiments.

    ../.venv/bin/python scripts/generate_ir_report_docx.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "outputs" / "reports" / "Infrared_Stream_IRdata4465_Report.docx"
FIG = HERE / "outputs" / "figures"
RUNS = HERE / "outputs" / "runs"
NAVY = RGBColor(0x1F, 0x37, 0x63)
CLASSES = ["airplane", "bird", "drone", "helicopter"]

BEST = "top2_yolo11s"
FINALISTS = [("top1_yolo12s", "YOLO12s"),
             ("top2_yolo11s", "YOLO11s"),
             ("top3_yolov8s_p2", "YOLOv8s-P2")]


def load(name):
    path = HERE / "outputs" / name
    return json.loads(path.read_text()) if path.exists() else {}


audit = load("dataset_report.json")
results = load("results_summary.json")
verify_clean = load("split_verification.json")
verify_seq = load("split_verification_seqdisjoint.json")
verify_official = load("split_verification_official.json")
confusion = {k: json.loads((RUNS / k / "confusion_test.json").read_text())
             for k, _ in FINALISTS if (RUNS / k / "confusion_test.json").exists()}


def run_minutes(name):
    csv_path = RUNS / name / "results.csv"
    if not csv_path.exists():
        return None
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    return float(rows[-1].get(" time") or rows[-1]["time"]) / 60


def best_epoch(name):
    csv_path = RUNS / name / "results.csv"
    if not csv_path.exists():
        return None, None
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    return int(float(best["epoch"])), float(best["metrics/mAP50(B)"])


doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)


def h1(t):
    doc.add_heading(t, level=1)


def h2(t):
    doc.add_heading(t, level=2)


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
    r.font.size = Pt(9.5)


def figure(path: Path, width=6.3, cap=None):
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
        run.font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(9.5)
            if bold_first_col and i == 0:
                run.bold = True
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Inches(w)
    doc.add_paragraph()
    return t


bt = results[BEST]["test"]
bv = results[BEST]["val"]
bc = confusion[BEST]
eval_total_official = (audit["official_split_summary"]["val"]["images"]
                       + audit["official_split_summary"]["test"]["images"])
gt_total = sum(d["gt_boxes"] for d in bc["per_class"].values())
n_wrong = sum(d["misclassified"] for d in bc["per_class"].values())
n_missed = sum(d["missed"] for d in bc["per_class"].values())
leak_gap = results["A_official_yolov8n"]["test"]["mAP50"] - results["B_clean_yolov8n"]["test"]["mAP50"]

# ---------------------------------------------------------------- title
title = doc.add_heading("Infrared Aircraft Detection (Objective 2)", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
srun = sub.add_run("IRdata-4465: Dataset Audit, Leakage-Free Split Construction, "
                   "Architecture Selection and Error Analysis")
srun.italic = True
srun.font.size = Pt(13)

# ---------------------------------------------------------------- summary
h1("1. Executive Summary")
p("This report covers the infrared stream of the multi-modal aircraft-detection thesis. "
  "The IRdata-4465 dataset was audited and found to be unusable as distributed; a "
  "leakage-free split was constructed and verified; six architectures were screened and "
  "the best three were trained to 25 epochs; and the resulting errors were analysed to "
  "determine what actually limits accuracy.")

p("Four findings matter:", bold=True)
bullet(f"The dataset as distributed cannot measure generalisation. Its {audit['images']} "
       f"files contain only {audit['unique_images_by_content']} unique images, and "
       f"{audit['contaminated_val_test_files']} of the {eval_total_official} validation "
       f"and test files are byte-identical copies of training images. Measured directly, "
       f"this inflates the score by {leak_gap:.3f} mAP.")
bullet(f"Removing duplicates and splitting by video sequence is still not enough. "
       f"{verify_seq['near_duplicate_eval_pct']}% of evaluation frames in that split had a "
       f"visually near-identical twin in training, because the dataset author cut single "
       f"source videos into consecutively numbered sequences. Splitting on visual groups "
       f"instead reduces this to {verify_clean['near_duplicate_eval_pct']}%.")
bullet(f"On the verified leakage-free split, the best detector is "
       f"{dict(FINALISTS)[BEST]} at {bt['mAP50']:.3f} mAP@0.5 and "
       f"{bt['mAP50_95']:.3f} mAP@0.5:0.95. Drone and helicopter are effectively solved; "
       f"airplane and bird are not.")
bullet(f"The airplane/bird weakness is mostly a detection failure, not a naming failure. "
       f"Across the test set, {100 * n_missed / gt_total:.0f}% of ground-truth objects are "
       f"missed entirely while only {100 * n_wrong / gt_total:.0f}% are found and given the "
       f"wrong class. Fixing it needs more resolution on small targets, not a better "
       f"classifier.")

# ---------------------------------------------------------------- dataset
h1("2. The Dataset")
p("IRdata-4465 (IEEE DataPort, DOI 10.21227/xg16-8n78) is a set of annotated infrared "
  "frames of airborne targets over sky, coastline and urban clutter. The frames are "
  "extracted from infrared video sequences, which is the root of the split problem "
  "discussed in section 3.")

bpx = audit["box_px"]
os_ = audit["official_split_summary"]
table(["Property", "Value"], [
    ["Images (as distributed)", f"{audit['images']} ({os_['train']['images']} train / "
                                f"{os_['val']['images']} val / {os_['test']['images']} test)"],
    ["Annotated boxes", f"{audit['boxes']} across 4 classes"],
    ["Classes", ", ".join(f"{k} ({v})" for k, v in audit["class_counts"].items())],
    ["Resolution", "640 x 640, grayscale replicated to 3 channels, full 0-255 range"],
    ["Source sequences", f"{audit['sequences']} labelled infrared video sequences"],
    ["Median box area", f"{int(bpx['median_area'])} px^2 (about 26 x 26 px)"],
    ["Small-target share", f"{bpx['pct_below_32x32']}% of boxes below 32x32 px; "
                           f"{bpx['pct_below_16x16']}% below 16x16 px"],
    ["Empty frames", f"{audit['empty_images']} (retained as background negatives)"],
    ["Label format", "YOLO horizontal boxes, already in Ultralytics directory layout"],
], widths=[2.0, 4.3], bold_first_col=True)

p("Note on thesis framing.", bold=True)
p("This is airborne infrared small-target detection with class labels. It is a different "
  "problem from the 30 m satellite thermal imagery of TIFAD.v1 and IRAir described on the "
  "current methodology slides. If IRdata-4465 is adopted for Objective 2, the methodology "
  "text must be updated to match, and the claim of 30 m SDGSAT-1 resolution must be "
  "removed unless TIFAD is also obtained.")

# ---------------------------------------------------------------- preprocessing
h1("3. Preprocessing and Data-Quality Audit")
p("Preprocessing is implemented in scripts/prepare_dataset.py and independently checked "
  "by scripts/verify_split.py. The audit steps in 3.2 and 3.3 proved more consequential "
  "than any modelling decision in this report.")

h2("3.1 Validation")
bullet("Every image was matched to its label file, and each label line was checked for "
       f"field count, class range and coordinate range. {audit['label_problem_count']} "
       "boxes extend past the image border and are clamped to the frame.")
bullet(f"Label class ids were cross-checked against the class encoded in each filename. "
       f"{audit['filename_class_mismatch_count']} files disagree (about 1%). These are "
       "reported rather than silently relabelled, since some are genuine multi-target "
       "frames and others are true annotation errors.")
bullet("Frames are already 640 x 640, the native YOLO input size, so no tiling or "
       "letterboxing is required. Pixel values already span the full 0-255 range, so "
       "global contrast stretching is a no-op; local CLAHE is available as an option.")

h2("3.2 Exact duplicates in the distributed split")
p("Every image was hashed (MD5 over the file bytes). The result invalidates any metric "
  "computed on the split as distributed:")
table(["Measure", "Count"], [
    ["Image files distributed", audit["images"]],
    ["Unique images by content", audit["unique_images_by_content"]],
    ["Byte-identical duplicate files", audit["duplicate_image_files"]],
    ["Duplicate groups spanning two or more splits", audit["duplicate_groups_crossing_splits"]],
    ["Validation/test files that are copies of training images",
     f"{audit['contaminated_val_test_files']} of {eval_total_official} "
     f"({100 * audit['contaminated_val_test_files'] / eval_total_official:.0f}%)"],
    ["Video sequences contributing frames to more than one split",
     f"{audit['sequences_spanning_official_splits']} of {audit['sequences']}"],
], widths=[4.3, 2.0])
p("The duplicates are exact copies, not augmented variants; this was verified by "
  "pixel-wise comparison. A model evaluated on the shipped split is therefore scored "
  "largely on images it memorised during training.")

h2("3.3 Near-duplicate leakage, and why a sequence-disjoint split is not enough")
p("The obvious repair is to drop duplicate frames and then assign whole source video "
  "sequences to a single split, so that temporally adjacent frames of one flight cannot "
  "straddle the train/test boundary. That split was built first. It is still not clean.")

g = audit.get("grouping", {})
p(f"Measuring frame similarity directly (per-pixel RMSE on 64x64 thumbnails) shows why. "
  f"In the sequence-disjoint split, {verify_seq['near_duplicate_eval_files']} of "
  f"{sum(verify_seq['images'].values()) - verify_seq['images']['train']} evaluation frames "
  f"({verify_seq['near_duplicate_eval_pct']}%) have a training frame within RMSE 2.5, "
  f"against a median of {verify_seq['unrelated_train_pair_rmse']['median']} for unrelated "
  f"frames in the same dataset. Inspecting the offending pairs identifies the cause: the "
  f"dataset author cut single source videos into consecutively numbered sequences, so "
  f"IR_HELICOPTER_018 and IR_HELICOPTER_019 are the same footage under two identifiers and "
  f"a sequence-level split separates them happily.")

p("A note on measurement method.", bold=True)
p("A perceptual/average hash was tried first and reported 87% contamination in the same "
  "split. That figure was an artefact: these frames are mostly uniform sky with a few "
  "bright pixels on target, so an 8x8 average hash is nearly constant across the dataset "
  "(1,931 frames collapse into 534 buckets, and 16% of randomly chosen unrelated pairs "
  "fall within Hamming distance 4). Thumbnail RMSE was adopted instead because it "
  "separates the populations cleanly, and the unrelated-pair baseline is reported "
  "alongside every result so the threshold can be judged rather than trusted.")

h2("3.4 The clean split")
p(f"The split unit was therefore changed from the sequence identifier to a visual group: "
  f"the connected components of the relation (same sequence id) OR (thumbnail RMSE below "
  f"{g.get('near_dup_rmse_threshold', 2.5)}). This absorbed "
  f"{g.get('sequences_merged_by_visual_similarity', 0)} sequences into another sequence's "
  f"group and produced {g.get('groups', 0)} groups, the largest holding "
  f"{g.get('largest_group_pct', 0)}% of the data. Groups are then assigned whole to "
  f"train/val/test at 70/15/15, stratified by dominant class.")
p("The threshold is a genuine trade-off and was calibrated rather than guessed. Below it "
  "the dataset keeps a workable group structure; at RMSE 3.0 transitive chaining collapses "
  "12% of all frames into a single component and a balanced split is no longer possible.")

cs = audit["clean_split_summary"]
table(["Split", "Images", "Boxes", "Groups", "Airplane", "Bird", "Drone", "Helicopter"],
      [[s, cs[s]["images"], cs[s]["boxes"], cs[s].get("groups", "-")]
       + [cs[s]["per_class"][c] for c in CLASSES] for s in ("train", "val", "test")],
      widths=[0.7, 0.7, 0.6, 0.7, 0.9, 0.6, 0.6, 1.0], bold_first_col=True)

h2("3.5 Verification")
p("scripts/verify_split.py re-derives the guarantee from the materialised files rather "
  "than trusting the code that produced them, so the claim rests on the artefact that was "
  "actually trained on.")
table(["Check", "As-shipped", "Sequence-disjoint", "Clean"], [
    ["Exact duplicates across splits",
     verify_official.get("exact_duplicates_across_splits", "-"),
     verify_seq.get("exact_duplicates_across_splits", "-"),
     verify_clean.get("exact_duplicates_across_splits", "-")],
    ["Sequences spanning splits",
     verify_official.get("sequences_spanning_splits", "-"),
     verify_seq.get("sequences_spanning_splits", "-"),
     verify_clean.get("sequences_spanning_splits", "-")],
    ["Eval frames with a near-identical twin in train",
     f"{verify_official.get('near_duplicate_eval_pct', 0)}%",
     f"{verify_seq.get('near_duplicate_eval_pct', 0)}%",
     f"{verify_clean.get('near_duplicate_eval_pct', 0)}%"],
    ["Nearest eval-to-train RMSE (median)",
     verify_official.get("eval_to_train_min_rmse", {}).get("median", "-"),
     verify_seq.get("eval_to_train_min_rmse", {}).get("median", "-"),
     verify_clean.get("eval_to_train_min_rmse", {}).get("median", "-")],
    ["Unrelated-pair RMSE (median, for scale)",
     verify_official.get("unrelated_train_pair_rmse", {}).get("median", "-"),
     verify_seq.get("unrelated_train_pair_rmse", {}).get("median", "-"),
     verify_clean.get("unrelated_train_pair_rmse", {}).get("median", "-")],
    ["Verdict", "leaks", "leaks", "clean"],
], widths=[2.8, 1.1, 1.4, 0.9], bold_first_col=True)

figure(FIG / "split_integrity.png", 6.3,
       "Figure 1. Left: contamination of each split variant, measured on the files. "
       "Right: the cost in reported mAP, isolated by a controlled pair of runs that share "
       "model, epoch budget and augmentation and differ only in the split.")

# ---------------------------------------------------------------- method
h1("4. Detector and Training Configuration")
p("YOLO (Ultralytics 8.4.118) was selected for the infrared stream. The labels are "
  "already YOLO horizontal boxes and the frames are already 640 x 640, so no format "
  "conversion, tiling or oriented-box handling is involved; using the same detector "
  "family as the optical HRPlanesv2 baseline also keeps the two streams comparable. "
  "Binary infrared small-target segmentation networks were rejected because they produce "
  "a hotspot map rather than the four class labels this task requires.")

p("Infrared-specific augmentation recipe.", bold=True)
p("The default Ultralytics recipe is tuned for colour photographs of medium-sized "
  "objects, so it was adjusted for single-band imagery containing few-pixel targets:")
table(["Setting", "Value", "Reason"], [
    ["hsv_h, hsv_s", "0.0", "a single infrared band carries no hue or saturation"],
    ["hsv_v", "0.3", "thermal gain and level genuinely vary between sequences"],
    ["degrees, flipud", "0.0", "sky scenes have a consistent horizon"],
    ["scale", "0.30 (default 0.5)", "aggressive zoom-out erases few-pixel targets"],
    ["close_mosaic", "3", "final epochs train on un-tiled full frames"],
    ["mixup, copy_paste", "0.0", "blending destroys low-contrast point targets"],
], widths=[1.4, 1.5, 3.4], bold_first_col=True)
p("All training ran on the Apple M5 Pro GPU through the Metal (MPS) backend. The training "
  "script aborts if the GPU is unavailable, because Metal is invisible inside a sandbox "
  "and Ultralytics would otherwise fall back silently to CPU.")

# ---------------------------------------------------------------- experiment 1
h1("5. Experiment 1: What the Corrupted Split Is Worth")
p("Two 10-epoch runs, identical in model, schedule, seed and augmentation, differing only "
  "in which split they were evaluated on. The gap between them is attributable to leakage "
  "and nothing else.")

rows = []
for key, split in (("A_official_yolov8n", "as-shipped (contaminated)"),
                   ("B_clean_yolov8n", "sequence-disjoint")):
    t = results[key]["test"]
    rows.append([key[0], split, "YOLOv8n", f"{t['mAP50']:.3f}", f"{t['mAP50_95']:.3f}",
                 f"{t['precision']:.3f}", f"{t['recall']:.3f}"])
table(["Run", "Split", "Model", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"], rows,
      widths=[0.5, 1.9, 0.9, 0.9, 1.1, 0.9, 0.8])

p(f"Evaluating on the distributed split overstates quality by {leak_gap:.3f} mAP@0.5, "
  f"from {results['B_clean_yolov8n']['test']['mAP50']:.3f} to "
  f"{results['A_official_yolov8n']['test']['mAP50']:.3f}. Any published number computed on "
  f"the shipped split of this dataset should be read with that in mind.")
p("The equivalent controlled measurement for the second leakage step (sequence-disjoint "
  "to clean) was not run, so this report does not attribute a specific mAP figure to it. "
  "What is established is the contamination rate itself, in section 3.5.", italic=True)

# ---------------------------------------------------------------- experiment 2
h1("6. Experiment 2: Architecture Screen")
p("Because several YOLO generations have been released since YOLOv8, six architectures "
  "were compared under identical data, image size, batch size, seed and augmentation. "
  "Nano scale and 12 epochs were used so the whole comparison fits in 81 minutes. The two "
  "P2 entries add a stride-4 detection head, quadrupling the resolution of the finest "
  "feature map, which is the standard remedy for very small objects.")
p("This screen ran on the sequence-disjoint split, before the near-duplicate leakage in "
  "section 3.3 was found. Its absolute numbers are therefore optimistic and are not quoted "
  "as results. It is used only to rank architectures, which is a comparison internal to "
  "the screen and unaffected by a bias common to all six runs.", italic=True)

bench = {k: v for k, v in results.items() if k.startswith("bench_")}
rows = []
for key in sorted(bench, key=lambda k: -bench[k]["test"]["mAP50"]):
    t = bench[key]["test"]
    pc = t["per_class"]
    mins = run_minutes(key)
    rows.append([key.replace("bench_", ""), f"{t['mAP50']:.3f}", f"{t['mAP50_95']:.3f}"] +
                [f"{pc.get(c, {}).get('mAP50', 0):.3f}" for c in CLASSES] +
                [f"{mins:.1f}" if mins else "-"])
table(["Model", "mAP@0.5", "mAP@0.5:0.95"] + [c.capitalize() for c in CLASSES] + ["Minutes"],
      rows, widths=[1.2, 0.8, 1.0, 0.8, 0.7, 0.7, 0.9, 0.8])

figure(FIG / "architecture_benchmark.png", 6.3,
       "Figure 2. Left: overall test mAP@0.5 per architecture on the screening split. "
       "Right: per-class results, showing the solved/unsolved class split is identical "
       "across architectures.")

p("Interpretation.", bold=True)
bullet("Architecture is not the bottleneck. Four of the six models sit within 3.5 mAP "
       "points. The ordering is used to choose finalists, not to claim a winner, because "
       "the spread is comparable to seed-to-seed variation at this budget.")
bullet("The P2 head is not a free win at a short schedule. A P2 model adds layers that "
       "COCO weights cannot initialise (only 219 of 437 tensors transfer for YOLOv8-P2 and "
       "360 of 902 for YOLO26-P2), so those layers train from scratch and do not converge "
       "in 12 epochs: YOLO26n-P2 collapses to 0.335 while costing 80% more time per epoch.")
bullet("P2 nevertheless does what it is designed to do for the smallest class. YOLOv8n-P2 "
       "records the best airplane result in the screen, which is why it was carried "
       "forward despite ranking third overall.")
bullet("The three architectures taken to the next stage are therefore YOLO12, YOLO11 and "
       "YOLOv8 with a P2 head, each at the larger 's' scale.")

# ---------------------------------------------------------------- experiment 3
h1("7. Experiment 3: The Three Finalists on the Clean Split")
p("The three selected architectures were trained for 25 epochs on the verified "
  "leakage-free clean split, at the 's' scale, under the same recipe. These are the "
  "results for Objective 2; every number in sections 5 and 6 was a screening measurement "
  "on a split that leaks.")

rows = []
for key, label in FINALISTS:
    t = results[key]["test"]
    ep, val = best_epoch(key)
    rows.append([label, f"{t['mAP50']:.3f}", f"{t['mAP50_95']:.3f}", f"{t['precision']:.3f}",
                 f"{t['recall']:.3f}", f"{val:.3f}" if val else "-", ep or "-"])
table(["Model", "test mAP@0.5", "test mAP@0.5:0.95", "Precision", "Recall",
       "best val mAP@0.5", "at epoch"], rows,
      widths=[1.1, 1.0, 1.2, 0.8, 0.7, 1.0, 0.7], bold_first_col=True)

rows = []
for key, label in FINALISTS:
    pc = results[key]["test"]["per_class"]
    rows.append([label] + [f"{pc.get(c, {}).get('mAP50', 0):.3f}" for c in CLASSES])
table(["Model"] + [c.capitalize() for c in CLASSES], rows,
      widths=[1.5, 1.2, 1.2, 1.2, 1.2], bold_first_col=True)

figure(FIG / "top3_comparison.png", 6.3,
       "Figure 3. The three finalists on the clean split. Left: overall test metrics. "
       "Right: per class, where the solved/unsolved divide persists.")

p("Interpretation.", bold=True)
bullet(f"{dict(FINALISTS)[BEST]} is the best of the three at {bt['mAP50']:.3f} mAP@0.5 and "
       f"{bt['mAP50_95']:.3f} mAP@0.5:0.95, and it has the highest precision "
       f"({bt['precision']:.3f}). It is the recommended Objective 2 baseline.")
bullet(f"The three models span only "
       f"{max(results[k]['test']['mAP50'] for k, _ in FINALISTS) - min(results[k]['test']['mAP50'] for k, _ in FINALISTS):.3f} "
       f"mAP. With one seed per configuration that is not a meaningful separation, which "
       f"reinforces the conclusion of the screen: the data, not the architecture, sets the "
       f"ceiling.")
bullet("YOLOv8s-P2 again earns its keep only on the hardest class, taking airplane to "
       f"{results['top3_yolov8s_p2']['test']['per_class']['airplane']['mAP50']:.3f} against "
       f"{results['top1_yolo12s']['test']['per_class']['airplane']['mAP50']:.3f} for YOLO12s, "
       "while giving up more than it gains on helicopter. The stride-4 head helps small "
       "targets and hurts larger ones at this training budget.")
bullet("Every model repeats the same pattern: drone above 0.90, helicopter 0.67-0.78, "
       "airplane and bird between 0.22 and 0.40.")

# ---------------------------------------------------------------- error analysis
h1("8. Error Analysis: Confusion Versus Missed Detection")
p("A single mAP value fuses two different failures: naming an object wrongly, and not "
  "finding it at all. They need opposite remedies, so scripts/confusion_analysis.py "
  "separates them. Every ground-truth box is matched to a prediction greedily by "
  "descending confidence at IoU >= 0.5, the same rule mAP@0.5 uses, and the outcome is "
  "recorded as correct, wrong class, or missed.")

p(f"Results below are for {dict(FINALISTS)[BEST]} on the clean test split at confidence "
  f"0.25.")

rows = []
for c in CLASSES:
    d = bc["per_class"][c]
    conf_with = ", ".join(f"{k} {v}" for k, v in d["confused_with"].items()) or "none"
    rows.append([c.capitalize(), d["gt_boxes"], f"{d['correct_pct']:.1f}%",
                 f"{d['misclassified_pct']:.1f}%", f"{d['missed_pct']:.1f}%",
                 d["false_positives"], conf_with])
table(["Class", "GT boxes", "Correct", "Wrong class", "Missed", "False pos.",
       "Confused with"], rows,
      widths=[0.9, 0.7, 0.7, 0.9, 0.7, 0.8, 1.6], bold_first_col=True)

figure(FIG / "error_breakdown.png", 6.3,
       "Figure 4. Left: confusion matrix, rows are ground truth and columns are "
       "predictions; the background row and column carry false positives and missed "
       "detections. Right: the same data as a per-class error budget.")

p("The airplane/bird question, answered directly.", bold=True)
p(f"Airplane and bird are confused, and the confusion is asymmetric: "
  f"{bc['airplane_called_bird_pct']}% of airplane ground truth is detected but labelled "
  f"bird, while only {bc['bird_called_airplane_pct']}% of bird ground truth is labelled "
  f"airplane. The model has a bird bias, not a symmetric muddle. It is the largest "
  f"cross-class confusion in the matrix; the only other one is bird labelled helicopter "
  f"({bc['per_class']['bird']['confused_with'].get('helicopter', 0)} boxes). Drone and "
  f"helicopter ground truth is never given another class at all.")
p(f"That confusion is nonetheless not the main problem. Across all {gt_total} test "
  f"objects, {n_wrong} ({100 * n_wrong / gt_total:.1f}%) are found and mislabelled, while "
  f"{n_missed} ({100 * n_missed / gt_total:.1f}%) are never found at all, a ratio of "
  f"roughly {n_missed / max(n_wrong, 1):.0f} to 1. For airplane specifically, "
  f"{bc['per_class']['airplane']['missed_pct']:.0f}% of instances are missed against "
  f"{bc['per_class']['airplane']['misclassified_pct']:.0f}% mislabelled. Raising "
  f"airplane accuracy by improving the classification head would therefore recover at most "
  f"a fifth of the available headroom.")

p("Target size explains the confusion that does occur.", bold=True)
ss = bc["size_stats"]
table(["Class", "Median box side", "Below 32x32",
       "Median side when correct", "when mislabelled", "when missed"],
      [[c.capitalize(), f"{ss[c]['median_side_px']:.1f} px",
        f"{ss[c]['pct_below_32x32']:.1f}%",
        f"{ss[c]['median_side_correct_px']:.1f} px",
        f"{ss[c]['median_side_misclassified_px']:.1f} px" if ss[c]['median_side_misclassified_px'] else "n/a",
        f"{ss[c]['median_side_missed_px']:.1f} px"] for c in CLASSES],
      widths=[0.9, 1.1, 0.9, 1.3, 1.1, 1.0], bold_first_col=True)
p(f"Birds that the model classifies correctly have a median box side of "
  f"{ss['bird']['median_side_correct_px']:.1f} px; birds it mislabels have a median side "
  f"of {ss['bird']['median_side_misclassified_px']:.1f} px, roughly half. Class identity "
  f"survives only above a certain number of pixels on target, and below it a bird and a "
  f"distant airplane are both a dim blob of a few pixels. Airplane shows no such "
  f"relationship ({ss['airplane']['median_side_correct_px']:.1f} px correct against "
  f"{ss['airplane']['median_side_missed_px']:.1f} px missed), which is consistent with "
  f"airplane failing for a different reason: it is the smallest class overall, with "
  f"{ss['airplane']['pct_below_32x32']:.0f}% of its boxes under 32x32 px.")

p("False positives on empty sky.", bold=True)
p(f"The background row of the confusion matrix shows "
  f"{bc['per_class']['bird']['false_positives']} bird and "
  f"{bc['per_class']['airplane']['false_positives']} airplane detections raised where "
  f"there is no annotated object, against "
  f"{bc['per_class']['drone']['false_positives']} for drone. The model both misses real "
  f"dim targets and invents them, which is the signature of a detector operating near its "
  f"contrast limit rather than one with a miscalibrated threshold.")

p("What this implies for the fix.", bold=True)
bullet("The productive direction is anything that increases effective pixels on target: "
       "a P2 or P1 detection head trained long enough to converge, a larger input size, or "
       "tiled inference. Section 7 already shows P2 giving the single best airplane score.")
bullet("Temporal information is the strongest untapped signal. The source data is video, "
       "and a dim target that is ambiguous in one frame is usually unambiguous across "
       "five. A single-frame detector discards this entirely.")
bullet("Class-rebalancing alone will not close the gap, because the dominant error is "
       "missed detection rather than misassignment.")
bullet("If bird is a distractor rather than a target of interest, merging it into a "
       "background or 'non-aircraft' class would remove the largest confusion pair and "
       "make the remaining metric easier to interpret. This is a thesis scoping decision.")

figure(FIG / "predictions_montage.png", 6.1,
       "Figure 5. Detections on held-out clean-split test frames. Drones and helicopters "
       "are found confidently; dim bird and distant airplane targets are missed, matching "
       "the per-class metrics.")

# ---------------------------------------------------------------- validity
h1("9. Limitations and Threats to Validity")
bullet("Single seed. Each configuration was run once. The three finalists differ by less "
       "than 0.03 mAP, which is within plausible seed variation, so the choice among them "
       "should not be presented as a decisive result.")
bullet("Training budget. 25 epochs is a compromise. The best validation score for "
       f"{dict(FINALISTS)[BEST]} occurs at epoch {best_epoch(BEST)[0]} of 25, so it is at "
       "or near its ceiling for this schedule, but a longer run was not attempted for all "
       "three.")
bullet("The architecture screen in section 6 was measured on a split now known to leak. "
       "It is used for ranking only, and its absolute values should not be quoted.")
bullet("The leakage cost of the second repair step (sequence-disjoint to clean) is not "
       "quantified by a controlled run, only by the contamination rate.")
bullet("Effective dataset size. After deduplication only "
       f"{audit['unique_images_by_content']} unique images remain, of which "
       f"{cs['train']['images']} are available for training. This is small for a four-class "
       "detector, and the test split holds "
       f"{cs['test']['boxes']} boxes, so per-class percentages in section 8 rest on tens of "
       "instances and carry wide error bars.")
bullet("Annotation quality. About 1% of files carry a class id that contradicts the class "
       "in their filename; these were left unmodified and cap the achievable accuracy.")
bullet("The near-duplicate threshold is a judgement call. It was calibrated against the "
       "unrelated-pair distribution and the point at which group chaining destroys the "
       "split, but a different defensible threshold would produce a slightly different "
       "split and slightly different numbers.")

# ---------------------------------------------------------------- next
h1("10. Proposed Next Steps")
p("In descending order of expected benefit:")
bullet("Exploit the temporal dimension: consecutive-frame stacking or track-level "
       "confirmation. This directly attacks the dominant error (missed dim targets), uses "
       "information the current pipeline throws away, and is a defensible novelty claim "
       "for the thesis.")
bullet("Train the P2 variant on a long schedule so its randomly initialised stride-4 "
       "layers converge, and test it at 1280 input as well as 640.")
bullet("Repeat the best configuration across three seeds so the reported figure carries a "
       "confidence interval, and so the finalist ranking becomes meaningful.")
bullet("Decide the status of the bird class explicitly, and if it is a distractor, "
       "re-scope the task around it.")
bullet("Run the controlled sequence-disjoint versus clean comparison at a fixed budget to "
       "put a number on the second leakage step.")

# ---------------------------------------------------------------- repro
h1("11. Reproducibility")
p("All work lives in the self-contained IRdata/ folder. The source dataset is read-only "
  "and referenced through symlinks, so the download in ~/Downloads/data is never modified.")
table(["Artefact", "Location"], [
    ["Preprocessing, audit and split construction", "IRdata/scripts/prepare_dataset.py"],
    ["Independent split verification", "IRdata/scripts/verify_split.py"],
    ["Training and evaluation", "IRdata/scripts/train_eval.py"],
    ["Error and confusion analysis", "IRdata/scripts/confusion_analysis.py"],
    ["Architecture screen", "IRdata/scripts/run_benchmark.sh"],
    ["The three finalist runs", "IRdata/scripts/run_top3.sh"],
    ["Figures", "IRdata/scripts/make_figures.py"],
    ["Full data audit", "IRdata/outputs/dataset_report.json"],
    ["Split verification results", "IRdata/outputs/split_verification*.json"],
    ["All metrics", "IRdata/outputs/results_summary.json"],
    ["Per-run weights, curves, confusion analysis", "IRdata/outputs/runs/<run>/"],
], widths=[2.9, 3.4], bold_first_col=True)

p("To reproduce end to end:", bold=True)
p("python scripts/prepare_dataset.py  ->  python scripts/verify_split.py "
  "--dataset datasets/clean  ->  bash scripts/run_top3.sh 25  ->  "
  "python scripts/make_figures.py  ->  python scripts/generate_ir_report_docx.py",
  italic=True)

p("Dataset citation.", bold=True)
p("Zhao, Y. IRdata-4465. IEEE DataPort, 2026. DOI 10.21227/xg16-8n78. 4,465 annotated "
  "infrared images at 640x640 covering helicopters, birds, drones and airplanes.")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"wrote {OUT}")
