"""What the shipped SARDet-100K split's leakage is worth on the aircraft class.

Two YOLOv8s runs, identical in everything except the split they were trained on:

    official_yolov8s   the split SARDet-100K ships
    clean_yolov8s      the group-disjoint split (no near-duplicate group spans two splits)

A near-duplicate "twin" is an image in the same visual group (thumbnail RMSE < 2.0,
see prepare_dataset.py). Each model is scored on the subsets below. Val is scored
as well as test because SARDet-100K results are commonly reported on val, and the
shipped val split carries most of the leak:

    official model  official val|test, all      the number a paper using the shipped split reports
    official model  official val|test, twin     images whose near-copy it trained on
    official model  official val|test, no twin  images it has never seen in any form
    clean model     clean test                  the number this project quotes
    both models     common unseen             official-test images that sit in the clean test
                                              split and have no twin in official train, so
                                              neither model saw them or a near-copy

Inflation of the reported number = (official test, all) - (official test, no twin).
The twin / no-twin gap shows how much memorised ground is worth per image, and the
common-unseen pair asks whether training on the leaky split changes how well the
model generalises, rather than only how it scores.

    ../.venv/bin/python scripts/leakage_ab.py --prepare-only   # CPU: groups + subsets
    ../.venv/bin/python scripts/leakage_ab.py                  # GPU: evaluate (outside sandbox)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "outputs" / "leakage"
SUBSETS = HERE / "datasets" / "leakage"
GROUPS_CACHE = OUT / "visual_groups.json"


def load_groups():
    """name -> (official split, visual group), computed once and cached."""
    if GROUPS_CACHE.exists():
        return json.loads(GROUPS_CACHE.read_text())
    from prepare_dataset import scan, visual_groups

    print("scanning annotations and grouping near-duplicates (reads every image) ...")
    records = scan()
    stats = visual_groups(records)
    names = [r["name"] for r in records]
    if len(set(names)) != len(names):
        raise SystemExit("file names repeat across splits; subsets would collide")
    groups = {"stats": stats,
              "images": {r["name"]: [r["official_split"], int(r["group"])] for r in records}}
    OUT.mkdir(parents=True, exist_ok=True)
    GROUPS_CACHE.write_text(json.dumps(groups))
    return groups


def build_subsets():
    groups = load_groups()["images"]
    train_groups = {g for split, g in groups.values() if split == "train"}

    evals = [n for n, (s, _) in groups.items() if s in ("val", "test")]
    leaked = sum(1 for n in evals if groups[n][1] in train_groups)
    print(f"official val+test images with a train twin: {leaked} / {len(evals)} "
          f"({100.0 * leaked / len(evals):.1f}%)")

    clean_split = {p.stem: s for s in ("train", "val", "test")
                   for p in (HERE / "datasets" / "clean" / "images" / s).iterdir()}
    clean_test = {k for k, s in clean_split.items() if s == "test"}

    subsets = {}
    for split in ("val", "test"):
        present = {p.stem for p in (HERE / "datasets" / "official" / "images" / split).iterdir()}
        stems = {Path(n).stem: n for n, (s, _) in groups.items() if s == split}
        stems = {k: v for k, v in stems.items() if k in present}
        no_twin = sorted(k for k, n in stems.items() if groups[n][1] not in train_groups)
        subsets[f"official_{split}_twin"] = ("official", split, sorted(
            k for k, n in stems.items() if groups[n][1] in train_groups))
        subsets[f"official_{split}_no_twin"] = ("official", split, no_twin)
        if split == "test":
            subsets["common_unseen"] = ("official", split, [k for k in no_twin if k in clean_test])

    # Scene level (overlapping crops of one SAR scene), from overlap_audit.py.
    scene_file = OUT / "scene_groups.json"
    if scene_file.exists():
        scene = {Path(n).stem: g for n, g in json.loads(scene_file.read_text()).items()}
        split_of = {Path(n).stem: s for n, (s, _) in groups.items()}
        for variant, assign in (("official", split_of), ("clean", clean_split)):
            train_scenes = {scene[k] for k, s in assign.items() if s == "train" and k in scene}
            for split in ("val", "test") if variant == "official" else ("test",):
                ev = sorted(k for k, s in assign.items() if s == split and k in scene)
                subsets[f"{variant}_{split}_scene_clean"] = (
                    variant, split, [k for k in ev if scene[k] not in train_scenes])
                subsets[f"{variant}_{split}_scene_shared"] = (
                    variant, split, [k for k in ev if scene[k] in train_scenes])
        clean_scene_unseen = set(subsets["clean_test_scene_clean"][2])
        subsets["common_scene_unseen"] = ("official", "test", [
            k for k in subsets["official_test_scene_clean"][2] if k in clean_scene_unseen])

    manifest = {"twin_rmse_threshold": 2.0, "subsets": {}}
    for name, (variant, split, keep) in subsets.items():
        root = SUBSETS / name
        src = HERE / "datasets" / variant
        if root.exists():
            shutil.rmtree(root)
        (root / "images" / "test").mkdir(parents=True)
        (root / "labels" / "test").mkdir(parents=True)
        boxes = 0
        for stem in keep:
            src_img = src / "images" / split / f"{stem}.jpg"
            src_lbl = src / "labels" / split / f"{stem}.txt"
            (root / "images" / "test" / f"{stem}.jpg").symlink_to(src_img.resolve())
            shutil.copyfile(src_lbl, root / "labels" / "test" / f"{stem}.txt")
            boxes += sum(1 for line in src_lbl.read_text().splitlines() if line.strip())
        (root / "data.yaml").write_text(
            f"path: {root.resolve()}\ntrain: images/test\nval: images/test\n"
            f"test: images/test\nnc: 1\nnames: ['aircraft']\n")
        manifest["subsets"][name] = {"from": f"{variant} {split}", "images": len(keep),
                                     "boxes": boxes}
        print(f"  {name:<28} {len(keep):>5} images  {boxes:>6} boxes")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subsets.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def evaluate(device: str, batch: int):
    from ultralytics import YOLO

    runs = HERE / "outputs" / "runs"
    models = {m: runs / f"{m}_yolov8s" / "weights" / "best.pt" for m in ("official", "clean")}
    for m, w in models.items():
        if not w.exists():
            raise SystemExit(f"missing {w}; train it with run_splits.sh first")

    official_yaml = HERE / "datasets" / "official" / "data.yaml"
    plan = [("clean", "clean test", HERE / "datasets" / "clean" / "data.yaml", "test")]
    for split in ("val", "test"):
        plan += [
            ("official", f"official {split}, all", official_yaml, split),
            ("official", f"official {split}, twin",
             SUBSETS / f"official_{split}_twin" / "data.yaml", "test"),
            ("official", f"official {split}, no twin",
             SUBSETS / f"official_{split}_no_twin" / "data.yaml", "test"),
        ]
    plan += [(m, "common unseen", SUBSETS / "common_unseen" / "data.yaml", "test")
             for m in ("official", "clean")]
    scene_level = (SUBSETS / "clean_test_scene_clean" / "data.yaml").exists()
    if scene_level:
        for model_name, split in (("official", "val"), ("official", "test"), ("clean", "test")):
            for part in ("scene_clean", "scene_shared"):
                plan.append((model_name, f"{model_name} {split}, {part.replace('_', ' ')}",
                             SUBSETS / f"{model_name}_{split}_{part}" / "data.yaml", "test"))
        plan += [(m, "common scene unseen", SUBSETS / "common_scene_unseen" / "data.yaml", "test")
                 for m in ("official", "clean")]

    rows = []
    for model_name, subset, yaml, split in plan:
        res = YOLO(str(models[model_name])).val(
            data=str(yaml), split=split, imgsz=512, batch=batch, device=device,
            plots=False, project=str(OUT / "val"),
            name=f"{model_name}__{subset.replace(', ', '_').replace(' ', '_')}", exist_ok=True)
        images = sum(1 for _ in (yaml.parent / "images" / split).iterdir())
        rows.append({"model": model_name, "scored_on": subset, "images": images,
                     "mAP50": round(float(res.box.map50), 4),
                     "mAP50_95": round(float(res.box.map), 4),
                     "precision": round(float(res.box.mp), 4),
                     "recall": round(float(res.box.mr), 4)})
        print(rows[-1])

    def get(model, subset, key):
        return next(r[key] for r in rows if r["model"] == model and r["scored_on"] == subset)

    summary = {}
    for key in ("mAP50", "mAP50_95"):
        summary[key] = {"common_unseen_official_minus_clean": round(
            get("official", "common unseen", key) - get("clean", "common unseen", key), 4)}
        for split in ("val", "test"):
            summary[key][split] = {
                "inflation_of_reported_number": round(
                    get("official", f"official {split}, all", key)
                    - get("official", f"official {split}, no twin", key), 4),
                "twin_minus_no_twin": round(
                    get("official", f"official {split}, twin", key)
                    - get("official", f"official {split}, no twin", key), 4),
                "reported_official_minus_quoted_clean": round(
                    get("official", f"official {split}, all", key)
                    - get("clean", "clean test", key), 4),
            }
        if scene_level:
            full = {("official", "val"): "official val, all", ("official", "test"): "official test, all",
                    ("clean", "test"): "clean test"}
            for (model_name, split), whole in full.items():
                summary[key][f"{model_name}_{split}_scene"] = {
                    "inflation_of_reported_number": round(
                        get(model_name, whole, key)
                        - get(model_name, f"{model_name} {split}, scene clean", key), 4),
                    "shared_minus_clean": round(
                        get(model_name, f"{model_name} {split}, scene shared", key)
                        - get(model_name, f"{model_name} {split}, scene clean", key), 4),
                }
            summary[key]["common_scene_unseen_official_minus_clean"] = round(
                get("official", "common scene unseen", key)
                - get("clean", "common scene unseen", key), 4)
    report = {"subsets": json.loads((OUT / "subsets.json").read_text()),
              "results": rows, "summary": summary}
    (OUT / "leakage_ab.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nreport -> {OUT / 'leakage_ab.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    build_subsets()
    if not args.prepare_only:
        evaluate(args.device, args.batch)


if __name__ == "__main__":
    main()
