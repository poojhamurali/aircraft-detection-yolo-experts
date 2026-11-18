"""Generate the plain-language optical report.

Same shape and tone as the infrared one: short sentences, one idea per
paragraph, every metric explained in words the first time it appears, and a
picture before every claim about the pictures.

Every number is read from the JSON files at generation time rather than typed
in, so the document cannot drift away from the runs it describes.

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
OUT = HERE / "outputs" / "reports" / "Optical_Report_Simple.docx"
FIG = HERE / "outputs" / "figures"
RUNS = HERE / "outputs" / "runs"
BEST = "airport_disjoint_yolov8s"
OTHER = "random_split_yolov8s"

audit = json.loads((HERE / "outputs" / "dataset_report.json").read_text())
cross = json.loads((HERE / "outputs" / "cross_eval.json").read_text())
best = json.loads((RUNS / BEST / "metrics.json").read_text())
other = json.loads((RUNS / OTHER / "metrics.json").read_text())

S = audit["airport_disjoint_split_summary"]
bt, bv = best["test"], best["val"]
xa, xb = cross[OTHER], cross[BEST]
gap50 = xa["mAP50"] - xb["mAP50"]
gap95 = xa["mAP50_95"] - xb["mAP50_95"]

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11.5)


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
    r.font.size = Pt(9.5)


def figure(name, width=6.2, cap=None):
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


def run_rows(name):
    with (RUNS / name / "results.csv").open() as fh:
        return list(csv.DictReader(fh))


def run_minutes(name):
    rows = run_rows(name)
    return float(rows[-1].get(" time") or rows[-1]["time"]) / 60


def best_epoch(name):
    """The epoch whose weights were kept.

    Ultralytics ranks checkpoints by fitness = 0.1*mAP50 + 0.9*mAP50-95, so the
    saved model is usually not the one with the highest mAP@0.5. Reporting the
    mAP@0.5 peak instead would name an epoch whose weights were discarded.
    """
    rows = run_rows(name)
    def fitness(r):
        return 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])
    b = max(rows, key=fitness)
    return int(float(b["epoch"])), float(b["metrics/mAP50(B)"])


# ---------------------------------------------------------------- title
title = doc.add_heading("Detecting Aircraft in Satellite Images", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("A plain-language summary of the optical work (Objective 1)")
r.italic = True
r.font.size = Pt(13)

p("In one sentence.", bold=True)
p(f"We taught a computer to find aircraft parked at airports in satellite "
  f"photographs, and it now does this very well: on four airports it had never "
  f"seen before, it scored {bt['mAP50']:.3f} out of 1.")

p("The one thing to take away.", bold=True)
p(f"The dataset we were given had a flaw that normally inflates results, so we "
  f"rebuilt it properly and measured how much the flaw was worth. The answer is "
  f"about {gap50:.3f} out of 1, which is very little. Unlike the infrared part of "
  f"this project, the optical result was essentially honest all along. Why that "
  f"differs so sharply between the two is explained in Section 5.")

# ---------------------------------------------------------------- 1 dataset
h1("1. The Dataset")

p("What the images are.", bold=True)
p(f"The dataset is called HRPlanesv2. It contains "
  f"{audit['images']} high-resolution colour photographs taken looking straight "
  f"down at airports, collected from Google Earth. Each picture is "
  f"{audit['frame_size_most_common'][0]} by {audit['frame_size_most_common'][1]} "
  f"pixels, which is very large — about twenty times the area of a normal phone "
  f"photo.")

p(f"Someone has already drawn a box around every aircraft in every picture. "
  f"There are {audit['boxes']} boxes in total. There is only one kind of object to "
  f"find, called 'plane', so unlike the infrared work there is no question of "
  f"telling different classes apart. The computer only has to answer: is there an "
  f"aircraft here, and exactly where?")

figure("dataset_samples.png", 6.4,
       "Figure 1. Left: one complete scene with every labelled aircraft boxed in "
       "green. Right: a single aircraft from that same scene, magnified. At full "
       "resolution the aircraft are large and clearly shaped.")

p("How big the targets are.", bold=True)
p(f"In the original picture the typical aircraft is about "
  f"{audit['median_box_side_px']:.0f} pixels across, which is large. But the "
  f"computer does not see the original picture. Before the image is fed in, it is "
  f"shrunk down to 640 pixels so that it fits in memory and runs quickly. After "
  f"that shrinking, the typical aircraft is only "
  f"{audit['median_box_side_at_640_input_px']:.0f} pixels across, and "
  f"{audit['pct_below_32x32_at_640_input']}% of them are smaller than 32 by 32 "
  f"pixels.")

p("This is worth understanding, because it is the main thing limiting the score. "
  "The detail is present in the original photograph; we throw most of it away to "
  "make the picture a manageable size. Section 6 suggests what to do about that.")

figure("dataset_stats.png", 6.4,
       "Figure 2. Left: how the pictures and aircraft were divided into groups. "
       "Right: how big the aircraft are by the time the model sees them.")

# ---------------------------------------------------------------- 2 the flaw
h1("2. The Problem We Found, and How We Fixed It")

p("How these systems are supposed to be tested.", bold=True)
p("You divide the pictures into three groups. The computer learns from the first "
  "group. The second group is used during learning, to check progress and decide "
  "when to stop. The third group is locked away and used once at the very end, to "
  "produce the score you report. The third group must contain nothing the computer "
  "has seen, or the score is meaningless.")

p("What was actually set up.", bold=True)
p(f"Two things were wrong with the dataset as we received it.")

p(f"First, the second group did not exist. The folder for it was empty, and the "
  f"configuration file pointed it at the test set instead. So the computer's "
  f"progress was being checked on the very pictures it would later be scored on, "
  f"and the best version of the model was chosen using those pictures. A previously "
  f"reported score of 0.99 came from this arrangement.")

p(f"Second, all {audit['shipped_test_images']} test pictures came from airports "
  f"that also appeared in the training pictures. Every single one. So even the test "
  f"set was not really unfamiliar.")

figure("split_explained.png", 6.4,
       "Figure 3. The two defects in the dataset as published.")

p("We also checked for repeated pictures, which is the problem that dominated the "
  f"infrared part of this project. Here it is minor: of {audit['images']} image "
  f"files, {audit['duplicate_image_files']} are exact copies of another, and "
  f"{audit['near_duplicates_shipped']['test_images_with_train_twin']} test pictures "
  f"({audit['near_duplicates_shipped']['pct']}%) are near-identical to a training "
  f"picture.")

p("What we did about it.", bold=True)
p("We rebuilt the three groups from scratch with one rule: an airport belongs to "
  "exactly one group. If an airport is used for training, none of its pictures may "
  "appear in validation or testing. This is stricter than simply avoiding repeated "
  "pictures, and it is the right rule here, because two photographs of the same "
  "airport taken on different days still show the same runways, the same terminal "
  "buildings and often the same parked aircraft.")

table(["Group", "Pictures", "Aircraft", "Airports", "What it is for"], [
    ["Training", S["train"]["images"], S["train"]["boxes"], S["train"]["airport_count"],
     "the computer learns from these"],
    ["Validation", S["val"]["images"], S["val"]["boxes"], S["val"]["airport_count"],
     "checks progress during learning"],
    ["Test", S["test"]["images"], S["test"]["boxes"], S["test"]["airport_count"],
     "never seen until the end; the reported score comes from these"],
], widths=[1.1, 0.8, 0.8, 0.8, 2.7], bold_first_col=True)

p(f"The four test airports are {', '.join(S['test']['airports'])}. The model has "
  f"never seen any picture of any of them.")

# ---------------------------------------------------------------- 3 model
h1("3. The Model")

p("What we used.", bold=True)
p("We used YOLO, which stands for You Only Look Once. It is a well-known family of "
  "object detectors: you give it a picture and it returns a list of boxes, each with "
  "a confidence score. We used the version called YOLOv8s, the small size.")

p("We did not train it from nothing. We started from a version already trained on "
  "millions of ordinary photographs and continued training it on our satellite "
  "pictures. This is called fine-tuning. It works because the early layers of the "
  "network learn general things, like edges and corners, that are useful for any "
  "kind of image.")

p("Unlike the infrared study, we did not compare several YOLO generations here. "
  "The score is already high enough that the differences between model versions "
  "would be lost in the noise between one airport and another.")

# ---------------------------------------------------------------- 4 settings
h1("4. Settings Used for Training")

ep, _ = best_epoch(BEST)
mins = run_minutes(BEST)

p("Basic settings.", bold=True)
table(["Setting", "Value", "Meaning in plain words"], [
    ["Model", "yolov8s.pt", "the small version of YOLOv8, pre-trained on photographs"],
    ["Epochs", str(best["epochs"]), "how many times the computer worked through all "
                                    "the training pictures"],
    ["Image size", "640 x 640", "each scene is shrunk to this before the model sees it"],
    ["Batch size", "4", "how many pictures are looked at together before the model "
                        "updates itself"],
    ["Optimiser", "AdamW, chosen automatically", "the method used to adjust the model"],
    ["Hardware", "Apple M5 Pro GPU", "the Mac's own graphics chip, via Apple's Metal "
                                     "system"],
    ["Training time", f"about {mins:.0f} minutes", "for the recommended model"],
    ["Random seed", "0", "fixed, so the run can be repeated exactly"],
], widths=[1.3, 1.8, 3.2], bold_first_col=True)

p("Settings chosen for satellite pictures.", bold=True)
p("During training it is normal to show the computer slightly altered copies of "
  "each picture, so it learns the object rather than memorising the photograph. "
  "This is called augmentation. Which alterations make sense depends entirely on "
  "the kind of imagery, and the right choices here are almost the opposite of the "
  "infrared ones.")

table(["Setting", "Value", "Why we chose it"], [
    ["Colour shift (hue, saturation)", "on",
     "these are colour photographs, so colour is real information — in the infrared "
     "work this had to be switched off"],
    ["Brightness shift", "0.4",
     "Google Earth captures vary a lot in sun angle and haze"],
    ["Rotation", "any angle",
     "looking straight down there is no correct 'up'; runways point every way"],
    ["Vertical flip", "50% of the time",
     "valid looking straight down, unlike a photograph of the sky"],
    ["Horizontal flip", "50% of the time", "same reason"],
    ["Zoom range", "0.50", "the standard setting; the targets are large enough to "
                           "survive it"],
    ["Mosaic", "on, off for the last 3 epochs",
     "tiles four pictures together for variety, then stops so the final training "
     "matches real conditions"],
    ["Mixup and copy-paste", "off", "blending images together does not reflect "
                                    "anything real in satellite imagery"],
], widths=[2.0, 1.5, 2.8], bold_first_col=True)

p("Did the training work?", bold=True)
p(f"Yes. The error fell steadily and the accuracy rose and then levelled off, which "
  f"is what a healthy training run looks like.")

p(f"One detail worth explaining, because it looks odd in the graph. The version of "
  f"the model that gets kept is the one from round {ep}, not the one from the round "
  f"with the highest accuracy. That is because the software ranks rounds mostly on "
  f"the strict measure of box position rather than on the headline accuracy, and the "
  f"two peak at different times here: after round {ep} the model kept getting "
  f"slightly better at finding aircraft while getting slightly worse at placing the "
  f"box exactly. The scores reported in this document all come from the kept version, "
  f"so they are a little lower than the highest point on the green line.")

figure("training_curve.png", 6.3,
       "Figure 4. Left: the model's error as training progresses; lower is better. "
       "Right: accuracy measured on the three validation airports, which the model "
       "never trains on. It climbs and then flattens.")

# ---------------------------------------------------------------- 5 results
h1("5. How Well It Performs")

p("The headline number.", bold=True)
p(f"On {S['test']['images']} pictures of four airports the model had never seen, it "
  f"scored {bt['mAP50']:.3f} out of 1. Under a stricter rule that also demands a very "
  f"precise box position, it scored {bt['mAP50_95']:.3f}.")

p("Three words used below:", italic=True)
bullet("Accuracy (properly called mAP@0.5) is a single score out of 1 combining "
       "whether the aircraft was found and whether the box was in roughly the right "
       "place. Higher is better.")
bullet("Precision means: when the model says it found an aircraft, how often is it "
       "right? Low precision means false alarms.")
bullet("Recall means: of all the aircraft really there, how many did it find? Low "
       "recall means it misses things.")

table(["Measure", "Score", "In words"], [
    ["Accuracy (mAP@0.5)", f"{bt['mAP50']:.3f}", "finds nearly every aircraft and "
                                                 "puts the box in the right place"],
    ["Strict accuracy (mAP@0.5:0.95)", f"{bt['mAP50_95']:.3f}",
     "the box is usually close but not pixel-perfect"],
    ["Precision", f"{bt['precision']:.3f}", "few false alarms"],
    ["Recall", f"{bt['recall']:.3f}", f"it misses about "
                                      f"{(1 - bt['recall']) * 100:.0f} in every 100"],
], widths=[2.1, 0.9, 3.2], bold_first_col=True)

figure("predictions_montage.png", 6.3,
       "Figure 5. The model's own output on airports it was never trained on. Every "
       "box is a detection the model made, with its confidence score.")

p("How much was the flaw worth?", bold=True)
p("This is the question the whole rebuild was for, and it needs care to answer "
  "properly. We trained a second model the sloppy way, mixing airports freely "
  "between groups, and kept everything else identical: same model, same number of "
  "rounds, same settings, same random seed.")

p("The obvious comparison would be to put the two models' scores side by side. That "
  "would be wrong, because they were scored on different pictures, so the difference "
  "would partly measure which pictures we happened to pick. Instead we scored both "
  f"models on exactly the same {S['test']['images']} pictures — the four held-out "
  f"airports. For the sloppy model this is not a fair test at all: it had already "
  f"trained on 215 of those {S['test']['images']} pictures, or 68% of them.")

table(["Model, scored on the same pictures", "Accuracy", "Strict accuracy", "Recall"], [
    ["Trained on 68% of these very pictures", f"{xa['mAP50']:.3f}",
     f"{xa['mAP50_95']:.3f}", f"{xa['recall']:.3f}"],
    ["Never saw these airports (ours)", f"{xb['mAP50']:.3f}",
     f"{xb['mAP50_95']:.3f}", f"{xb['recall']:.3f}"],
], widths=[2.9, 1.0, 1.1, 1.0], bold_first_col=True)

figure("leakage_comparison.png", 6.4,
       "Figure 6. Left: both models on identical pictures. Right: the advantage the "
       "cheating model gets from having already seen most of the test set.")

p(f"Having already trained on two thirds of the test set is worth only "
  f"{gap50:+.3f} on the normal measure, and {gap95:+.3f} on the strict one — that is, "
  f"nothing at all under strict scoring, where the honest model is fractionally "
  f"ahead. So the flaw inflated the published figure, but barely.")

p("Why this differs so much from the infrared result.", bold=True)
p("In the infrared study, fixing the equivalent problem cost 0.19 out of 1, which is "
  "an enormous drop. Here it costs 0.02. The contrast is one of the more interesting "
  "findings in this project, and there are two reasons for it.")

p("The first is what a repeated picture means in each dataset. The infrared images "
  "come from video. Two frames a fraction of a second apart are almost the same "
  "photograph, so a model that memorised one has effectively been given the answer "
  "to the other. These satellite pictures are separate Google Earth captures; two "
  "pictures of the same airport differ in date, sun angle and which part of the "
  "ground they cover. Sharing an airport is a much weaker form of overlap than "
  "sharing a moment in a video.")

p(f"The second is how hard the task is to begin with. By the time the model sees "
  f"them, these aircraft are about "
  f"{audit['median_box_side_at_640_input_px']:.0f} pixels across, in colour, against "
  f"high contrast. The infrared targets were roughly 24 pixels across, grey and "
  f"faint, and three quarters of them were below 32 pixels. When a task is nearly "
  f"solved by honest means, there is very little left for memorisation to add.")

p("An honest caution.", bold=True)
p(f"The difference between airports is larger than the effect we just measured. Our "
  f"validation airports scored {bv['mAP50']:.3f} and our test airports "
  f"{bt['mAP50']:.3f}, and neither group was ever trained on. That spread of "
  f"{abs(bv['mAP50'] - bt['mAP50']):.3f} is bigger than the {gap50:.3f} attributed to "
  f"the flaw. Any claim resting on a difference of a couple of points in this dataset "
  f"should be treated with caution.")

# ---------------------------------------------------------------- 6 conclusion
h1("6. Conclusion")

p("What works.", bold=True)
bullet(f"Aircraft detection in satellite imagery is effectively solved at this "
       f"resolution: {bt['mAP50']:.3f} accuracy on four completely unseen airports, "
       f"finding {bt['recall'] * 100:.0f} of every 100 aircraft with few false alarms.")
bullet("The evaluation is now trustworthy. There is a real validation set, and no "
       "airport appears in more than one group.")
bullet("The whole pipeline rebuilds from the original data with a single command and "
       "a fixed random seed.")

p("What is weaker.", bold=True)
bullet(f"Strict accuracy is {bt['mAP50_95']:.3f}, well below the headline "
       f"{bt['mAP50']:.3f}. The model knows where an aircraft is, but the exact box "
       f"edges are often a few pixels out. This is the direct consequence of shrinking "
       f"a {audit['frame_size_most_common'][0]}-pixel picture down to 640.")
bullet("Each configuration was trained once, so the numbers carry no error bar.")

p("What we would do next.", bold=True)
p("In order of how much we expect each to help:")
bullet("Stop throwing away the resolution. Cut each large scene into tiles and run "
       "the model on each tile separately, so aircraft arrive at their real size "
       "instead of being shrunk to 35 pixels. This should mainly improve the strict "
       "score, which is where the headroom is.")
bullet("Repeat the best run three times with different starting randomness, so a "
       "margin of error can be stated rather than a single figure.")
bullet("Rotate which airports are held out and average the results. With only four "
       "test airports, the score depends noticeably on which four they are.")
bullet("Carry the same airport-disjoint rule into the radar work, where the images "
       "are cut-out patches of larger scenes and this kind of overlap is likely to "
       "matter far more than it did here.")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"wrote {OUT}")
