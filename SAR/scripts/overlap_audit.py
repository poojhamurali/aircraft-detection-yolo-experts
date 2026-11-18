"""Find SAR patches that are overlapping crops of the same scene.

SARDet-100K cut each large SAR-AIRcraft image into 512x512 patches with a 200 px
overlap (stride 312). Two horizontally adjacent patches therefore share a 512x200
strip of identical ground, yet as whole images they differ by far more than the
near-duplicate threshold in prepare_dataset.py, so that grouping cannot see them.

For every 512x512 patch, its right strip (columns 312-511) is compared with every
other patch's left strip (columns 0-199), and its bottom strip with every top
strip. Candidates come from 8x-downsampled strips; each candidate is then checked
at full resolution, where real overlaps agree pixel for pixel (only JPEG noise
differs) and unrelated strips do not, because speckle never repeats. Strips that
are nearly flat are skipped: dark, featureless ground matches anything.

Overlap links are unioned with the existing near-duplicate groups into scene
groups, and every split is audited against them.

    ../.venv/bin/python scripts/overlap_audit.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import Union  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "outputs" / "leakage"
SIZE, OVERLAP = 512, 200
DOWN = 8
MIN_STRIP_STD = 6.0     # grey levels, on the downsampled strip
CANDIDATE_RMSE = 6.0    # downsampled strips
CONFIRM_RMSE = 6.0      # full resolution; see the unrelated-pair baseline in the report


def image_paths(names):
    roots = [HERE / "raw" / "extracted" / s for s in ("train", "val", "test")]
    out = {}
    for n in names:
        for r in roots:
            if (r / n).exists():
                out[n] = r / n
                break
    return out


@lru_cache(maxsize=1024)
def grey(path: str):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def strips(img):
    """left, right, top, bottom strips; vertical ones transposed to the same shape."""
    return (img[:, :OVERLAP], img[:, SIZE - OVERLAP:],
            img[:OVERLAP, :].T, img[SIZE - OVERLAP:, :].T)


def small(strip):
    return cv2.resize(strip, (strip.shape[1] // DOWN, strip.shape[0] // DOWN),
                      interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def best_matches(A, B, valid_a, valid_b, block=2048):
    """For each row of A, the nearest row of B by RMSE (excluding the same index)."""
    sqa, sqb = (A * A).sum(1), (B * B).sum(1)
    idx = np.full(len(A), -1)
    dist = np.full(len(A), np.inf, dtype=np.float32)
    for s in range(0, len(A), block):
        e = min(s + block, len(A))
        d2 = sqa[s:e, None] + sqb[None, :] - 2.0 * (A[s:e] @ B.T)
        r = np.sqrt(np.maximum(d2, 0) / A.shape[1])
        r[np.arange(e - s), np.arange(s, e)] = np.inf
        r[:, ~valid_b] = np.inf
        j = r.argmin(1)
        idx[s:e], dist[s:e] = j, r[np.arange(e - s), j]
    idx[~valid_a] = -1
    return idx, dist


def main():
    groups = json.loads((OUT / "visual_groups.json").read_text())["images"]
    paths = image_paths(groups)
    names, feats = [], [[], [], [], []]
    for n, p in paths.items():
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape != (SIZE, SIZE):
            continue
        names.append(n)
        for k, s in enumerate(strips(im)):
            feats[k].append(small(s))
    print(f"{len(names)} 512x512 patches (SAR-AIRcraft)")
    L, R, T, B = (np.stack(f) for f in feats)
    flat = {k: X.std(1) >= MIN_STRIP_STD for k, X in zip("LRTB", (L, R, T, B))}

    links, baseline = [], []
    rng = np.random.default_rng(0)
    for (src, dst, Xs, Xd, ks, kd) in (("right", "left", R, L, 1, 0),
                                       ("bottom", "top", B, T, 3, 2)):
        j, d = best_matches(Xs, Xd, flat["LRTB"[ks]], flat["LRTB"[kd]])
        cand = np.flatnonzero((j >= 0) & (d < CANDIDATE_RMSE))
        confirmed = 0
        for i in cand:
            a = strips(grey(str(paths[names[i]])))[ks].astype(np.float32)
            b = strips(grey(str(paths[names[j[i]]])))[kd].astype(np.float32)
            rmse = float(np.sqrt(((a - b) ** 2).mean()))
            if rmse < CONFIRM_RMSE and groups[names[i]][1] != groups[names[j[i]]][1]:
                links.append((names[i], names[j[i]], src, round(rmse, 2)))
                confirmed += 1
        for _ in range(500):
            a_i, b_i = rng.integers(0, len(names), 2)
            a = strips(grey(str(paths[names[a_i]])))[ks].astype(np.float32)
            b = strips(grey(str(paths[names[b_i]])))[kd].astype(np.float32)
            baseline.append(float(np.sqrt(((a - b) ** 2).mean())))
        print(f"  {src}->{dst}: {len(cand)} candidates, {confirmed} confirmed overlaps")

    # scene groups = near-duplicate groups joined by overlap links
    all_names = list(groups)
    pos = {n: i for i, n in enumerate(all_names)}
    uf = Union(len(all_names))
    by_group = {}
    for n, (_, g) in groups.items():
        if g in by_group:
            uf.union(pos[by_group[g]], pos[n])
        else:
            by_group[g] = n
    for a, b, _, _ in links:
        uf.union(pos[a], pos[b])
    scene = {n: uf.find(pos[n]) for n in all_names}

    clean = {}
    for split in ("train", "val", "test"):
        for p in (HERE / "datasets" / "clean" / "images" / split).iterdir():
            clean[p.stem] = split

    def audit(split_of):
        train_scenes = {scene[n] for n in all_names if split_of(n) == "train"}
        out = {}
        for split in ("val", "test"):
            ev = [n for n in all_names if split_of(n) == split]
            hit = sum(1 for n in ev if scene[n] in train_scenes)
            out[split] = {"images": len(ev), "share_scene_with_train": hit,
                          "pct": round(100.0 * hit / max(len(ev), 1), 1)}
        return out

    sizes = Counter(scene.values())
    report = {
        "patches_checked": len(names),
        "overlap_links": len(links),
        "link_rmse_median": float(np.median([l[3] for l in links])) if links else None,
        "unrelated_strip_rmse": {"median": round(float(np.median(baseline)), 1),
                                 "p1": round(float(np.percentile(baseline, 1)), 1)},
        "scene_groups": len(sizes),
        "largest_scene_group": max(sizes.values()),
        "official_split": audit(lambda n: groups[n][0]),
        "clean_split": audit(lambda n: clean.get(Path(n).stem)),
        "thresholds": {"min_strip_std": MIN_STRIP_STD, "candidate_rmse": CANDIDATE_RMSE,
                       "confirm_rmse": CONFIRM_RMSE, "downsample": DOWN},
    }
    (OUT / "overlap_audit.json").write_text(json.dumps(report, indent=2))
    (OUT / "overlap_links.json").write_text(json.dumps(links))
    (OUT / "scene_groups.json").write_text(json.dumps(scene))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
