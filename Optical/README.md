# Optical stream (O1) — HRPlanesv2

Aircraft detection in nadir RGB satellite imagery. One class, `plane`.

This directory is self-contained: it reads `../dataset` and writes only inside
`Optical/`. Images are symlinked, never copied or modified.

## What was wrong with the dataset as shipped

`../dataset/data.yaml` declared `val: images/test`, and `images/val/` held zero
files. Validation *was* the test set, so the checkpoint was selected on the same
images it was then scored on. A previously reported mAP@0.5 of 0.99 came from
that arrangement and is not a generalisation result.

The audit (`scripts/prepare_dataset.py`, written to `outputs/dataset_report.json`)
also found:

| Finding | Count |
|---|---|
| Images | 2120 (2114 unique by content) |
| Boxes | 14333, median 266 px across in the native 4800×2703 frame (35.5 px at the model's 640 input; 41.7% under 32×32 there) |
| Airports (source groups) | 31 |
| Airports appearing in more than one shipped split | 26 of 31 |
| Shipped test images from an airport also in training | 424 of 424 (100%) |
| Byte-identical duplicates crossing shipped splits | 4 groups |
| Shipped test images with a near-duplicate training twin | 14 of 421 (3.3%) |

Near-duplicate detection uses 64×64 thumbnail RMSE, as in the infrared stream.
The median nearest-neighbour RMSE is 30.8 against an unrelated-pair baseline of
53.0, so the flagged pairs are genuinely close rather than an artefact of the
threshold.

## Splits built here

`scripts/prepare_dataset.py` writes two variants. Airports are parsed from the
filename (`12_DT5`, `DAV_032`), and the disjoint split uses greedy largest-first
bin-packing because one airport alone holds roughly 29% of the data.

| Variant | Train | Val | Test | Purpose |
|---|---|---|---|---|
| `random_split` | 1480 img / 10042 box / 31 airports | 317 / 2033 / 28 | 317 / 2206 / 30 | comparison only — airports mixed |
| `airport_disjoint` | 1480 / 9816 / 24 airports | 317 / 3067 / 3 (DT8, HON, SZX) | 317 / 1398 / 4 (DAL, DEN, JOH, SUV) | **quote this one** |

## Results

YOLOv8s, 25 epochs, 640 px, batch 4, seed 0, identical augmentation. The only
difference between the two runs is the split.

| Run | Val mAP@0.5 | Val mAP@0.5:0.95 | Test mAP@0.5 | Test mAP@0.5:0.95 |
|---|---|---|---|---|
| `random_split` | 0.9878 | 0.5840 | 0.9817 | 0.5667 |
| `airport_disjoint` | 0.9428 | 0.4928 | **0.9699** | **0.6259** |

**Headline: mAP@0.5 = 0.970 on four airports never seen during training.**

### What airport overlap is actually worth

The two rows above are scored on *different* images, so the difference between
them confounds leakage with the choice of test set. `scripts/cross_eval.py`
settles it by scoring both checkpoints on one common test set — the four unseen
airports — of which the random-split model trained on 215 of 317 images (67.8%)
outright.

| Model on the same 317 images | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| `random_split` (trained on 68% of them) | 0.9928 | 0.6247 | 0.9911 | 0.9857 |
| `airport_disjoint` (never saw them) | 0.9699 | 0.6259 | 0.9565 | 0.9434 |

Direct train-on-test of two thirds of the evaluation set buys **+0.023 mAP@0.5**,
and **nothing at all** at mAP@0.5:0.95 — the honest model is fractionally ahead
there. Leakage inflates this benchmark, but only slightly.

That is the opposite of the infrared stream, where removing leakage cost 0.189
mAP, and the contrast is the interesting result rather than either number alone.
Infrared frames come from video, so a leaked frame is nearly the same picture and
memorising it transfers directly to the test copy. HRPlanesv2 tiles are separate
Google Earth captures; two tiles of one airport differ in date, sun angle and
ground area, so sharing an airport is far weaker than sharing a scene. The
targets are also easier: at the 640 px input the model actually sees, the median
aircraft here is 35.5 px across with 41.7% under 32×32, against roughly 24 px and
75% for the infrared set, and they sit in colour at high contrast. An easy,
near-saturated task leaves little headroom for memorisation to exploit.

### Rotation augmentation cost most of the strict metric

`degrees=180` rotates each axis-aligned box and re-fits a larger axis-aligned box
around it, so training targets become looser than the aircraft. Retraining with
only that changed (`--degrees 0`, run `airport_disjoint_yolov8s_norot`):

| Run | Val mAP@0.5 | Val mAP@0.5:0.95 | Test mAP@0.5 | Test mAP@0.5:0.95 |
|---|---|---|---|---|
| `airport_disjoint_yolov8s` (±180°) | 0.9428 | 0.4928 | 0.9699 | 0.6259 |
| `airport_disjoint_yolov8s_norot` (0°) | 0.9815 | 0.7574 | **0.9931** | **0.8585** |

Rotation cost 0.23 mAP@0.5:0.95 on unseen airports, so the no-rotation run is the
optical number to quote. The leakage comparison above is unaffected: both of its
runs used the same augmentation.

### Which checkpoint is being reported

Ultralytics keeps the epoch with the best fitness, `0.1*mAP50 + 0.9*mAP50-95`,
which is dominated by the strict metric. For the airport-disjoint run that is
epoch 9 (val 0.9427 / 0.4929, matching `metrics.json` exactly), while val mAP@0.5
keeps climbing to 0.974 by epoch 19. After epoch 9 the model goes on getting
slightly better at finding aircraft while getting slightly worse at placing the
box. Quoting the mAP@0.5 peak would advertise a score no saved checkpoint
achieves, so the figures and the report both annotate epoch 9.

Note that airport-to-airport variation (val 0.943 vs test 0.970, both unseen)
is larger than the leakage effect. Any claim resting on a ~2 point difference in
this dataset should be treated with caution.

## Running it

```bash
cd Optical
../.venv/bin/python scripts/prepare_dataset.py      # build both splits
bash scripts/run_splits.sh 25 yolov8s.pt 4          # train both, ~70 min each
../.venv/bin/python scripts/cross_eval.py           # common-test-set comparison
../.venv/bin/python scripts/make_report_figures.py  # figures
../.venv/bin/python scripts/generate_report_docx.py # outputs/reports/Optical_Report_Simple.docx
```

`prepare_dataset.py --audit-only` refreshes `outputs/dataset_report.json` without
rebuilding `datasets/`, so the splits a trained model was scored on stay put.

The report reads every figure from the JSON files at generation time, so it
cannot drift from the runs it describes.

Training must run outside a sandbox. Metal is invisible to a sandboxed process
and Ultralytics then falls back to CPU silently; `train_eval.py` refuses to
start rather than let that happen.

## The MPS assigner bug

Training died repeatedly inside `TaskAlignedAssigner.get_box_metrics` with

```
RuntimeError: shape mismatch: value tensor of shape [3452]
cannot be broadcast to indexing result of shape [1983]
```

The assigner both gathers and scatters through one boolean mask, and on Metal
the two disagree about how many elements the mask selects. The element counts
differ from run to run, so this is a backend correctness bug, not a size limit:
batch 16 died in the first epoch and batch 4 died in the second. Reducing the
batch only changes how long it takes to hit.

`scripts/mps_tal_patch.py` computes the same two tensors by multiplying with the
mask rather than indexing by it, which is algebraically identical because both
destinations start as zeros. `scripts/test_mps_tal_patch.py` checks the
replacement against the original on CPU, where the original is correct, and the
two agree to 0.0e+00 on both outputs. The workaround evaluates IoU densely, and
batch 4 keeps the resulting intermediates bounded.

HRPlanesv2 provokes this and the infrared stream never did because these scenes
carry up to 127 aircraft each and mosaic tiles four scenes into one sample.

## A note on concurrency

`run_splits.sh` holds a lock recording its owner's pid and reclaims it only when
that process is genuinely gone. An earlier pid-less version could not distinguish
a live lock from a stale one; deleting a live one by hand let a second trainer
into the same run directory, where the two interleaved their `results.csv` and
destroyed each other's weights. For the same reason `train_eval.py` refuses to
write into a directory that already holds a `results.csv`.
