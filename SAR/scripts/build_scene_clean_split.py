"""The SAR split to quote: SARDet-100K's own split minus scene-level leaks.

overlap_audit.py showed that a random re-split of near-duplicate groups (the old
`clean` split) still puts overlapping crops of one scene on both sides: 34.7% of
its test images share a scene with its training set, worth +0.033 mAP@0.5. The
shipped split keeps scenes apart far better (1.4% of test, 12.5% of val), so it
is kept as is, and only the val/test images whose scene also appears in train
are removed. Train is unchanged, which keeps results comparable to published
SARDet-100K numbers.

    ../.venv/bin/python scripts/overlap_audit.py            # writes scene_groups.json
    ../.venv/bin/python scripts/build_scene_clean_split.py  # writes datasets/official_scene_clean
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "datasets" / "official"
DST = HERE / "datasets" / "official_scene_clean"


def main():
    scene = {Path(n).stem: g for n, g in
             json.loads((HERE / "outputs" / "leakage" / "scene_groups.json").read_text()).items()}
    train = sorted(p.stem for p in (SRC / "images" / "train").iterdir())
    train_scenes = {scene[k] for k in train}

    if DST.exists():
        shutil.rmtree(DST)
    report = {}
    for split in ("train", "val", "test"):
        (DST / "images" / split).mkdir(parents=True)
        (DST / "labels" / split).mkdir(parents=True)
        stems = sorted(p.stem for p in (SRC / "images" / split).iterdir())
        keep = stems if split == "train" else [k for k in stems if scene[k] not in train_scenes]
        for k in keep:
            (DST / "images" / split / f"{k}.jpg").symlink_to(
                (SRC / "images" / split / f"{k}.jpg").resolve())
            shutil.copyfile(SRC / "labels" / split / f"{k}.txt", DST / "labels" / split / f"{k}.txt")
        report[split] = {"images": len(keep), "removed_scene_shared": len(stems) - len(keep)}

    (DST / "data.yaml").write_text(
        f"path: {DST.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 1\nnames: ['aircraft']\n")
    (HERE / "outputs" / "leakage" / "official_scene_clean_split.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
