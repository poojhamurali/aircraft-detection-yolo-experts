# Aircraft Detection with YOLO: Optical, Thermal-Infrared and SAR Experts

One YOLO detector per sensor, each trained and evaluated on a train/test split that
has been audited for leakage. Part of the M.Tech project *Leakage-Audited Aircraft
Detection Across Optical, Thermal-Infrared and SAR Imagery* (Poojha Muralidharan,
Amrita School of Computing).

The single model that replaces these three experts lives in the companion repository
[unified-aircraft-detection](https://github.com/poojhamurali/unified-aircraft-detection).

## Streams

| Folder | Sensor | Dataset | Model | Test mAP@0.5 / mAP@0.5:0.95 |
|---|---|---|---|---|
| `Optical/` | Nadir RGB | HRPlanesv2 (31 airports) | YOLOv8s | 0.993 / 0.859 |
| `IRdata/` | Airborne thermal infrared | IRdata-4465 (372 videos) | YOLO11s | 0.584 / 0.306 (4 classes) |
| `SAR/` | Spaceborne SAR (C/X band) | SARDet-100K, aircraft class | YOLOv8s | 0.783 / 0.450 |
| `TIFAD/` | Spaceborne thermal (SDGSAT-1) | TIFAD.v1 | not trainable yet (scenes must be ordered) | – |

Each folder has its own README with the full audit, split construction, training
recipe and results.

## What the audit found

Every shipped split leaks, each through a different mechanism:

* **Optical:** all shipped test images come from airports also used in training, and
  the validation set was the test set. Rebuilt as an airport-disjoint split.
* **Thermal infrared:** 2,534 of 4,465 files are byte copies, and 83% of the shipped
  validation/test files are copies of training frames. Rebuilt by grouping video
  sequences and visually near-identical frames.
* **SAR:** SARDet-100K cuts large scenes into overlapping 512-pixel tiles.
  `SAR/scripts/overlap_audit.py` finds 8,637 pixel-identical tile overlaps; a
  de-duplicated random re-split still put 34.7% of test images in a training scene.
  The shipped split minus its scene-shared images is used instead.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training was run on an Apple-silicon GPU (PyTorch MPS backend). The scripts refuse to
fall back to the CPU silently; `mps_tal_patch.py` works around an MPS bug in the
Ultralytics task-aligned assigner.

## Data (not included)

Datasets and trained weights are not part of this repository. Place them as follows:

* **Optical:** HRPlanesv2 ([Zenodo](https://doi.org/10.5281/zenodo.7331974)) as
  `dataset/` at the repository root, then run `Optical/scripts/prepare_dataset.py`.
* **Thermal infrared:** the IRdata-4465 Roboflow export; pass its folder with
  `IRdata/scripts/prepare_dataset.py --src <folder>`.
* **SAR:** SARDet-100K under `SAR/raw/` (`SAR/scripts/fetch_chunked.py` downloads it),
  then run `SAR/scripts/prepare_dataset.py`.

Each stream's README lists the full sequence of commands.
