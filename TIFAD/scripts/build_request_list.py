"""Turn the TIFAD.v1 label filenames into an SDGSAT-1 scene request list.

TIFAD.v1 ships target chips and labels but not the parent scenes, so the scenes
have to be ordered separately from the SDGSAT-1 Open Science Program. Every label
filename encodes the scene it came from:

    KX10_TIS_20230305_E12.74_N42.04_202300055197_L4B_head_0.txt
    |    |   |        |       |      |            |    |
    sat  TIS date     lon     lat    scene/task   L4B  tile within scene

A fresh account carries a quota of only 20 scenes, so the list is sorted by the
number of annotated targets per scene: ordering the densest scenes first buys the
most labelled aircraft per unit of quota.

    python scripts/build_request_list.py
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LABELS = HERE / "raw" / "extracted" / "TIFAD.v1_label" / "TIFAD.v1_label"
OUT_DIR = HERE / "outputs"
QUOTA = 20  # default SDGSAT-1 allocation for a new account

NAME_RE = re.compile(
    r"^KX10_TIS_(?P<date>\d{8})_(?P<lon>[EW][\d.]+)_(?P<lat>[NS][\d.]+)"
    r"_(?P<scene>\d+)_L4B_head_(?P<head>\d+)$"
)


def signed(token: str) -> float:
    """E12.74 -> 12.74, W77.60 -> -77.60, S29.46 -> -29.46"""
    value = float(token[1:])
    return -value if token[0] in "WS" else value


def collect():
    scenes = defaultdict(lambda: {"tiles": 0, "targets": 0, "contrail": 0})
    unmatched = []
    for path in sorted(LABELS.glob("*.txt")):
        m = NAME_RE.match(path.stem)
        if not m:
            unmatched.append(path.name)
            continue
        classes = Counter()
        for line in path.read_text().splitlines():
            if line.strip():
                classes[line.split()[0]] += 1
        key = (m["scene"], m["date"], m["lon"], m["lat"])
        entry = scenes[key]
        entry["tiles"] += 1
        entry["targets"] += sum(classes.values())
        entry["contrail"] += classes.get("1", 0)
    return scenes, unmatched


def main():
    if not LABELS.is_dir():
        raise SystemExit(f"labels not found at {LABELS}; extract TIFAD.v1_label.7z first")

    scenes, unmatched = collect()
    rows = sorted(
        ([scene, date, lon, lat, signed(lon), signed(lat),
          v["tiles"], v["targets"], v["contrail"]]
         for (scene, date, lon, lat), v in scenes.items()),
        key=lambda r: (-r[7], r[0]),
    )
    header = ["scene_id", "date", "lon_token", "lat_token", "longitude", "latitude",
              "tiles", "annotated_targets", "contrail_targets"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full = OUT_DIR / "sdgsat1_scene_request_list.csv"
    with full.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    priority = OUT_DIR / f"sdgsat1_top{QUOTA}_request.csv"
    with priority.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows[:QUOTA])

    total = sum(r[7] for r in rows)
    quota_total = sum(r[7] for r in rows[:QUOTA])
    print(f"unmatched filenames : {len(unmatched)}")
    print(f"unique scenes       : {len(rows)}")
    print(f"annotated targets   : {total}")
    print(f"first {QUOTA} scenes cover : {quota_total} targets "
          f"({100 * quota_total / total:.1f}%)")
    print(f"\nfull list     -> {full}")
    print(f"priority list -> {priority}")

    print(f"\nsearch parameters for the {QUOTA} priority scenes "
          f"(sdgsat.ac.cn > Data Access):")
    print(f"  {'scene_id':<14}{'date':<10}{'lat':>8}{'lon':>9}{'targets':>9}")
    for r in rows[:QUOTA]:
        print(f"  {r[0]:<14}{r[1]:<10}{r[5]:>8.2f}{r[4]:>9.2f}{r[7]:>9}")


if __name__ == "__main__":
    main()
