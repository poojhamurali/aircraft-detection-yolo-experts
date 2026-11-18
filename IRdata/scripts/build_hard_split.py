"""Build `datasets/clean_hard`: the clean split with hard training frames oversampled.

Most test errors are airplanes and birds (missed, or named as each other), and the
training split holds only 249 airplane and 297 bird frames against 637 drone frames.
No new imagery is available for those backgrounds, so the training frames that exist
are re-weighted instead:

  * every training frame containing an airplane or a bird is repeated --class-repeat
    times in total;
  * training frames the baseline model gets wrong (a missed box, a wrong class or a
    false alarm at conf 0.25) get --hard-repeat further copies. Mining reads only the
    training split, so val and test stay untouched.

val and test are the clean split's own directories, symlinked, so scores remain
comparable with every other clean-split run. Copies are extra symlinks to the same
source file with a `repN__` prefix; nothing is copied or edited.

Usage:
    python scripts/build_hard_split.py --mine-run imp_base640_s0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "datasets" / "clean"
DST = HERE / "datasets" / "clean_hard"
HARD_CLASSES = {0, 1}  # airplane, bird


def label_classes(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {int(float(l.split()[0])) for l in path.read_text().splitlines() if l.strip()}


def mine_errors(run: str, device: str, conf: float) -> set[str]:
    """Training-frame stems where the baseline misses, misnames or hallucinates."""
    import numpy as np
    from ultralytics import YOLO

    from confusion_analysis import iou_matrix, load_gt

    model = YOLO(str(HERE / "outputs" / "runs" / run / "weights" / "best.pt"))
    images = sorted((SRC / "images" / "train").glob("*.jpg"))
    hard = set()
    for i in range(0, len(images), 16):
        batch = images[i:i + 16]
        for p, res in zip(batch, model.predict([str(x) for x in batch], conf=conf,
                                               device=device, verbose=False)):
            h, w = res.orig_shape
            gtb, gtc = load_gt(SRC / "labels" / "train" / f"{p.stem}.txt", w, h)
            pb = res.boxes.xyxy.cpu().numpy()
            pc = res.boxes.cls.cpu().numpy().astype(int)
            ious = iou_matrix(pb, gtb)
            matched_gt, ok = set(), True
            for pi in np.argsort(-res.boxes.conf.cpu().numpy()):
                cand = [g for g in range(len(gtb)) if g not in matched_gt and ious[pi, g] >= 0.5]
                if not cand:
                    ok = False  # false alarm
                    continue
                g = max(cand, key=lambda k: ious[pi, k])
                matched_gt.add(g)
                ok &= pc[pi] == gtc[g]
            if len(matched_gt) < len(gtb):
                ok = False
            if not ok:
                hard.add(p.stem)
    return hard


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(src.resolve(), dst.parent), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-run", required=True)
    ap.add_argument("--class-repeat", type=int, default=2)
    ap.add_argument("--hard-repeat", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    hard = mine_errors(args.mine_run, args.device, args.conf)
    if DST.exists():
        shutil.rmtree(DST)

    counts = {"frames": 0, "class_copies": 0, "hard_copies": 0}
    for img in sorted((SRC / "images" / "train").glob("*.jpg")):
        lbl = SRC / "labels" / "train" / f"{img.stem}.txt"
        copies = 1
        if label_classes(lbl) & HARD_CLASSES:
            copies += args.class_repeat - 1
            counts["class_copies"] += args.class_repeat - 1
        if img.stem in hard:
            copies += args.hard_repeat
            counts["hard_copies"] += args.hard_repeat
        for k in range(copies):
            prefix = "" if k == 0 else f"rep{k}__"
            link(img, DST / "images" / "train" / f"{prefix}{img.name}")
            if lbl.exists():
                link(lbl, DST / "labels" / "train" / f"{prefix}{lbl.name}")
            counts["frames"] += 1
    for split in ("val", "test"):
        for kind in ("images", "labels"):
            (DST / kind).mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.relpath(SRC / kind / split, DST / kind), DST / kind / split)

    (DST / "data.yaml").write_text(
        "# clean split, airplane/bird and baseline-error training frames oversampled\n"
        f"path: {DST}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 4\nnames: ['airplane', 'bird', 'drone', 'helicopter']\n")
    report = {"mine_run": args.mine_run, "mined_hard_frames": len(hard),
              "class_repeat": args.class_repeat, "hard_repeat": args.hard_repeat,
              "train_entries": counts["frames"], **counts}
    (HERE / "outputs" / "hard_split_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
