# SAR stream (O3) — SARDet-100K aircraft subset

Aircraft detection in Synthetic Aperture Radar patches. One class, `aircraft`.

This directory is self-contained. Images are extracted once into `raw/extracted/`
and then symlinked into `datasets/`; source files are never modified.

## Headline

**Test mAP@0.5 = 0.783, mAP@0.5:0.95 = 0.450** (YOLOv8s, 15 epochs) on
SARDet-100K's own split with the 42 test images that share a scene with training
removed (`datasets/official_scene_clean`, 2,998 test images).

The earlier headline of 0.851 came from the `clean` re-split, which turned out to
leak at the scene level (below). It is kept only as a leakage measurement.

Plain-language report: `outputs/reports/SAR_Report_Simple.docx` (predates the
scene-level audit; its numbers are the `clean` split's).

## Dataset

| | |
|---|---|
| Source | SARDet-100K (NeurIPS 2024), aircraft class only |
| Images with aircraft | 19,879 |
| Aircraft boxes | 52,669 |
| Unique by content | 13,268 (6,611 exact duplicates) |
| Sources | SAR-AIRcraft 18,888 · SADD 883 · MSAR 108 |
| Median box side | 72.4 px (23.7% under 32×32) |

## Two kinds of leakage

**Near-duplicates** (`prepare_dataset.py`): whole patches with thumbnail RMSE < 2.0.
4.6% of the shipped val+test images have one in train (201 val, 28 test).

**Overlapping crops of one scene** (`overlap_audit.py`): SARDet-100K cut each
SAR-AIRcraft image into 512 px patches with a 200 px overlap, so neighbouring
patches share a 512×200 strip pixel for pixel while differing as whole images.
Matching every patch's right/bottom strip against every left/top strip finds 8,637
such links (median strip RMSE 0.0, against 36.9 for unrelated strips). Joining them
with the near-duplicate groups gives scene groups:

| Split | Val sharing a scene with train | Test sharing a scene with train |
|---|---|---|
| shipped (`official`) | 246 / 1,972 (12.5%) | 42 / 3,040 (1.4%) |
| re-split (`clean`) | 1,068 / 2,982 (35.8%) | 1,034 / 2,982 (34.7%) |

The re-split removed near-duplicates but scattered the crops of each scene across
splits; the shipped split had kept them together.

## Results

From `scripts/leakage_ab.py` (`outputs/leakage/leakage_ab.json`):

| Model | Scored on | Images | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| official | official test, all | 3,040 | 0.787 | 0.455 |
| official | **official test, scene clean (quote this)** | **2,998** | **0.783** | **0.450** |
| official | official val, all | 1,972 | 0.928 | 0.594 |
| official | official val, scene clean | 1,726 | 0.925 | 0.588 |
| clean | clean test, all | 2,982 | 0.851 | 0.516 |
| clean | clean test, scene shared with train | 1,034 | 0.959 | 0.648 |
| clean | clean test, scene clean | 1,948 | 0.818 | 0.479 |
| official | test scenes neither model trained on | 263 | 0.774 | 0.418 |
| clean | test scenes neither model trained on | 263 | 0.867 | 0.526 |

- Scene leakage inflated the `clean` test score by +0.033 mAP@0.5; images whose scene
  was in training score +0.14 higher.
- The shipped test split's leaks are worth only +0.004.
- Even on scenes neither model trained on, the re-split model scores +0.09 higher.
  The strip matcher only finds exact 200 px neighbours, and SAR-AIRcraft covers few
  airports, so the re-split most likely still shares airports between train and test.
  SARDet-100K's renumbered files carry no airport names to test this directly.
- Official val scores 0.14 above official test even after removing detected overlaps,
  so SARDet-100K numbers reported on val should not be read as test performance.

The `clean` run was interrupted after epoch 14 and finished with
`scripts/resume_run.py` (same optimizer state and schedule), then re-scored with
`scripts/eval_checkpoint.py`.

## Running it

```bash
cd SAR
../.venv/bin/python scripts/fetch_chunked.py train   # if archives missing
../.venv/bin/python scripts/prepare_dataset.py
bash scripts/run_splits.sh 15 yolov8s.pt 8 official clean
../.venv/bin/python scripts/overlap_audit.py
../.venv/bin/python scripts/build_scene_clean_split.py
../.venv/bin/python scripts/leakage_ab.py            # GPU, outside the sandbox
```
