# IRdata-4465 — Infrared Aircraft / Drone Detection (Objective 2)

Self-contained workspace for the **infrared stream**. The source dataset (passed with
`--src`, `~/Downloads/data` by default) is read-only: images are symlinked, never
copied or edited.

**Headline:** on a verified leakage-free split, **YOLO11s reaches 0.584 mAP@0.5 /
0.306 mAP@0.5:0.95**. Getting to a split that measures anything real took more
work than the modelling did, and the remaining errors are dominated by *missed*
dim targets rather than by confusing one class for another.

## Dataset

**IRdata-4465** — 4,465 annotated infrared frames, 640x640, YOLO horizontal boxes.
Frames come from 372 infrared video sequences of airborne targets over sky,
coastline and urban clutter (`IR_AIRPLANE_013_frame_231`, `IR_DRONE_045_frame_125`, ...).

| Property | Value |
|---|---|
| Images | 4,465 (3,125 / 893 / 447 as shipped) |
| Boxes | 4,568 across 4 classes |
| Classes | 0 airplane (841), 1 bird (1,026), 2 drone (2,072), 3 helicopter (629) |
| Resolution | 640x640, grayscale replicated to 3 channels, full 0-255 range |
| Median box | ~26 x 26 px; 65% of boxes below 32x32 px; 11% below 16x16 px |
| Empty frames | 49 (kept as background negatives) |

This is **infrared small-target detection with class labels**, not the 30 m
satellite thermal problem of TIFAD.v1 (see `../TIFAD`).

## Preprocessing

`scripts/prepare_dataset.py` performs the audit and builds the trainable trees;
`scripts/verify_split.py` independently re-checks the result from the files on
disk rather than trusting the code that wrote them.

1. **Pair and label validation** — every image matched to its label file; field
   count, class range and coordinate range checked. 57 boxes cross the image
   border and are clamped to the frame.
2. **Class-map verification** — label ids cross-checked against the class encoded
   in each filename. 52 files (about 1%) disagree; they are left untouched and
   reported rather than silently relabelled.
3. **Duplicate detection (MD5 over image bytes)** — the 4,465 files contain only
   **1,931 unique images**. 2,534 files are byte-identical copies.
4. **Contamination measurement** — 840 duplicate groups span the shipped splits,
   so **1,109 of the 1,340 val+test files (83%) are exact copies of training
   images**. 362 of the 372 sequences contribute frames to more than one split.
5. **Near-duplicate grouping** — see below. This is the step that actually makes
   the split honest.
6. **Group-disjoint re-split** — 70/15/15, stratified by dominant class.
7. **Geometry / contrast** — frames are already 640x640, the native YOLO input,
   so no tiling or letterboxing. Pixels already span 0-255, so global stretching
   is a no-op; local CLAHE is available via `--clahe`.

### Why a sequence-disjoint split was not enough

The obvious repair is to drop duplicates and then assign whole source video
sequences to one split. That was built first (`datasets/seq_disjoint`) and it
**still leaks**: measuring frame similarity directly (per-pixel RMSE on 64x64
thumbnails) shows **176 of 559 evaluation frames (31.5%)** have a training frame
within RMSE 2.5, against a median of **60.8** for unrelated frames.

The cause is that the dataset author cut single source videos into consecutively
numbered sequences, so `IR_HELICOPTER_018` and `IR_HELICOPTER_019` are the same
footage under two identifiers, and a sequence-level split separates them happily.

The fix is to split on **visual groups**: connected components of
`(same sequence id) OR (thumbnail RMSE < 2.5)`. That absorbed **131 sequences**
into another sequence's group, leaving 241 groups with the largest holding 4.1%
of the data.

> A perceptual/average hash was tried first and reported 87% contamination. That
> was an artefact — these frames are mostly uniform sky, so an 8x8 average hash is
> near-constant across the dataset (1,931 frames collapse into 534 buckets, and
> 16% of *random unrelated* pairs fall within Hamming distance 4). Thumbnail RMSE
> separates the populations cleanly, and the unrelated-pair baseline is reported
> next to every result so the threshold can be judged rather than trusted.

The RMSE threshold is a real trade-off and was calibrated, not guessed: at 3.0,
transitive chaining collapses 12% of all frames into one component and a balanced
split becomes impossible.

### Split variants

| Variant | Images (train/val/test) | Eval frames with a twin in train | Use |
|---|---|---|---|
| `official` | 3,125 / 893 / 447 | 94.9% | as-shipped; contamination evidence only |
| `seq_disjoint` | 1,372 / 275 / 284 | 31.5% | superseded; screening runs only |
| **`clean`** | **1,389 / 300 / 242** | **0.0%** | **train and report on this** |

`clean` verification: 0 exact duplicates across splits, 0 sequences spanning
splits, 0 near-duplicate eval frames; nearest eval-to-train RMSE median 5.57
against an unrelated-pair median of 60.5.

## Algorithm

**YOLO (Ultralytics 8.4.118)**, fine-tuned from COCO weights — the same family as
the optical HRPlanesv2 baseline, so numbers are comparable across streams. Labels
are already YOLO horizontal boxes and 640x640 is the native input size, so no
format conversion or oriented boxes are involved. Binary infrared small-target
*segmentation* networks were rejected: they produce a hotspot map, not the four
class labels this task needs.

Infrared-specific training recipe (`IR_HYP` in `scripts/train_eval.py`):

- `hsv_h=0, hsv_s=0` — a single infrared band carries no hue or saturation
- `hsv_v=0.3` — thermal gain and level genuinely vary between sequences
- `degrees=0, flipud=0` — sky scenes have a consistent horizon
- `scale=0.30` (instead of 0.5) — aggressive zoom-out erases few-pixel targets
- `close_mosaic=3` — final epochs on un-tiled full frames
- `mixup=0, copy_paste=0` — blending destroys low-contrast point targets

All training runs on the Apple M5 Pro GPU (MPS). `train_eval.py` **aborts** if the
GPU is unavailable, because Metal is invisible inside a sandbox and Ultralytics
would otherwise fall back to CPU silently.

## Results — the three finalists (clean split, 25 epochs, 's' scale)

Six architectures were screened first at nano scale for 12 epochs; the top three
were carried forward. Screening ranked YOLO12 > YOLO11 > YOLOv8-P2, and YOLOv8-P2
was kept despite third place because it posted the best *airplane* score and is
the only one of the three that adds a detection level matched to these target
sizes.

| Model | test mAP@0.5 | mAP@0.5:0.95 | P | R | best val | at epoch | minutes |
|---|---|---|---|---|---|---|---|
| **YOLO11s** | **0.584** | **0.306** | 0.615 | 0.573 | 0.605 | 25 | 41.0 |
| YOLOv8s-P2 | 0.568 | 0.287 | 0.575 | 0.574 | 0.549 | 23 | 51.4 |
| YOLO12s | 0.555 | 0.275 | 0.560 | 0.631 | 0.588 | 20 | — * |

\* the YOLO12s wall-clock figure is not comparable; that run was contended by
other processes for its first ten epochs.

Per-class test mAP@0.5:

| Model | airplane | bird | drone | helicopter |
|---|---|---|---|---|
| YOLO11s | 0.241 | **0.403** | 0.928 | 0.763 |
| YOLOv8s-P2 | **0.335** | 0.355 | 0.908 | 0.674 |
| YOLO12s | 0.224 | 0.290 | **0.929** | **0.776** |

Reading of the numbers:

- **The three models span 0.03 mAP.** With one seed each that is not a meaningful
  separation. Consistent with the screen: the data, not the architecture, sets
  the ceiling.
- **Drone is solved** (>0.90) and **helicopter is close** (0.67-0.78). **Airplane
  and bird are not** (0.22-0.40), in every model.
- **P2 again earns its keep only on the hardest class**, taking airplane to 0.335
  against 0.224 for YOLO12s, while giving up more than it gains on helicopter.
- **Leakage was worth 0.189 mAP.** Same model, same budget, only the split
  differs: YOLOv8n scores 0.745 on the as-shipped split and 0.556 on the
  sequence-disjoint one.

## Error analysis: confusion vs missed detection

`scripts/confusion_analysis.py` matches every ground-truth box to a prediction
greedily by descending confidence at IoU >= 0.5 — the same rule mAP@0.5 uses — and
records the outcome as correct, wrong class, or missed. For YOLO11s on the clean
test split at confidence 0.25:

| Class | GT | Correct | Wrong class | Missed | False pos. | Confused with |
|---|---|---|---|---|---|---|
| airplane | 65 | 21.5% | 20.0% | 58.5% | 15 | bird 12, helicopter 1 |
| bird | 50 | 38.0% | 10.0% | 52.0% | 29 | helicopter 5 |
| drone | 92 | 92.4% | 0.0% | 7.6% | 11 | — |
| helicopter | 29 | 79.3% | 0.0% | 20.7% | 8 | — |

**Airplane and bird are confused, and it is asymmetric**: 18.5% of airplane ground
truth is detected but labelled bird, while 0% of bird ground truth is labelled
airplane. Drone and helicopter are never given another class.

**But confusion is not the main problem.** Across all 236 test objects, 18 (7.6%)
are found and mislabelled while 77 (32.6%) are never found at all — a ratio of
about 4 to 1. Improving the classification head could recover at most a fifth of
the available headroom.

**Target size explains the confusion that does occur.** Birds classified correctly
have a median box side of 42.5 px; birds mislabelled have a median side of 20.1 px,
roughly half. Class identity survives only above a certain number of pixels on
target; below it, a bird and a distant airplane are both a dim blob. Airplane
shows no such relationship (20.6 px correct vs 24.4 px missed) — it fails for a
different reason, being the smallest class overall with 75% of boxes under 32x32.

The model also raises 29 bird and 15 airplane detections on empty sky. Missing
real dim targets *and* inventing them is the signature of a detector at its
contrast limit, not a miscalibrated threshold.

**What this implies:** the productive direction is anything that increases
effective pixels on target (P2/P1 heads trained to convergence, larger input,
tiled inference), and above all **temporal information** — the source is video,
and a target that is ambiguous in one frame is usually unambiguous across five.
Class rebalancing alone will not close the gap.

## Reproducing

```bash
cd IRdata
../.venv/bin/python scripts/prepare_dataset.py                    # audit + build datasets/
../.venv/bin/python scripts/verify_split.py --dataset datasets/clean
bash scripts/run_top3.sh 25                                       # the three finalists
../.venv/bin/python scripts/collect_results.py
../.venv/bin/python scripts/make_figures.py                       # technical figures
../.venv/bin/python scripts/make_simple_figures.py                # plain-language figures
../.venv/bin/python scripts/generate_ir_report_docx.py            # technical report
../.venv/bin/python scripts/generate_simple_report_docx.py        # plain-language report
```

`run_top3.sh` must run **outside a sandbox** (Metal is otherwise invisible) and
**in the foreground** — backgrounded jobs are killed when their permission session
ends. It holds a lock so two invocations cannot collide, skips any run that
already has a `metrics.json`, and deletes an *unfinished* run directory rather
than training over it.

That last point matters. `train_eval.py` refuses to write into a directory that
already holds a `results.csv`, because two trainings sharing one directory
interleave their rows and corrupt both. An earlier version of `run_top3.sh`
passed `--force`, which disabled that guard for exactly the case it exists to
catch, and a `top2_yolo11s` run was duplicated into 36 rows for 25 epochs. Only
the training curve was affected — `metrics.json` is recomputed from `best.pt`
after training, and a clean retrain reproduced 0.5835 / 0.3056 exactly — but the
guard is now left intact.

Budget: roughly 1.5-2 hours for the three runs on an Apple M5 Pro GPU.

## Reports

| File | Audience |
|---|---|
| `outputs/reports/Infrared_Stream_IRdata4465_Report.docx` | technical: method, controls, threats to validity |
| `outputs/reports/Infrared_Report_Simple.docx` | plain language: dataset, model, settings, conclusion, with photos |

Both read every number from `outputs/*.json` at generation time, so neither can
drift from the experiments, and they cannot disagree with each other.

## Layout

```
IRdata/
  scripts/prepare_dataset.py       # audit, dedup, visual grouping, split, data.yaml
  scripts/verify_split.py          # independent leakage check on the built split
  scripts/train_eval.py            # YOLO training + val/test evaluation + predictions
  scripts/confusion_analysis.py    # correct / wrong-class / missed breakdown
  scripts/collect_results.py       # comparison table across runs
  scripts/make_figures.py          # technical figures
  scripts/make_simple_figures.py   # dataset photos + plain-language charts
  scripts/generate_ir_report_docx.py      # technical report
  scripts/generate_simple_report_docx.py  # plain-language report
  scripts/run_top3.sh              # the three finalist runs
  scripts/run_benchmark.sh         # six-architecture screen
  datasets/official/               # as-shipped split (contaminated)
  datasets/seq_disjoint/           # deduplicated, sequence-disjoint (still leaks)
  datasets/clean/                  # group-disjoint, verified leakage-free  <- use this
  outputs/dataset_report.json      # full preprocessing audit
  outputs/split_verification*.json # leakage checks per split variant
  outputs/results_summary.json     # collected metrics
  outputs/runs/<name>/             # weights, curves, metrics.json, confusion_test.json
  outputs/figures/                 # split_integrity, top3_comparison,
                                   # error_breakdown, predictions_montage,
                                   # dataset_samples, why_confused,
                                   # dataset_stats, simple_results, training_curve
  outputs/reports/                 # the two generated .docx reports
```
