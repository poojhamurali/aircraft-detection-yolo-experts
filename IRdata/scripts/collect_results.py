"""Collect metrics.json from every finished run into one comparison table."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CLASSES = ["airplane", "bird", "drone", "helicopter"]


def main():
    runs = sorted(p for p in (HERE / "outputs" / "runs").glob("*/metrics.json"))
    if not runs:
        raise SystemExit("no finished runs yet")

    rows = []
    for path in runs:
        m = json.loads(path.read_text())
        rows.append(m)

    header = f"{'run':26s} {'model':11s} {'test mAP50':>10s} {'mAP50-95':>9s} {'P':>6s} {'R':>6s}"
    print(header)
    print("-" * len(header))
    for m in rows:
        t = m["test"]
        print(f"{m['run']:26s} {m['model']:11s} {t['mAP50']:10.4f} {t['mAP50_95']:9.4f} "
              f"{t['precision']:6.3f} {t['recall']:6.3f}")

    print("\nper-class test mAP@0.5")
    print(f"{'run':26s} " + " ".join(f"{c:>11s}" for c in CLASSES))
    print("-" * (26 + 12 * len(CLASSES)))
    for m in rows:
        pc = m["test"]["per_class"]
        print(f"{m['run']:26s} " +
              " ".join(f"{pc.get(c, {}).get('mAP50', float('nan')):11.4f}" for c in CLASSES))

    summary = {m["run"]: m for m in rows}
    out = HERE / "outputs" / "results_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
