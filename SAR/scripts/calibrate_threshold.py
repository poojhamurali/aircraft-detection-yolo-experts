"""Pick the near-duplicate threshold from the data instead of guessing it.

SAR patches are mostly flat speckle, so two unrelated crops can look alike to a
crude descriptor. That is how the infrared perceptual hash went wrong. Before
trusting any near-duplicate count, compare the distance distribution for pairs
that might be duplicates against the distribution for pairs that certainly are
not, and choose a threshold where the second distribution is essentially absent.

    ../.venv/bin/python scripts/calibrate_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_dataset as P  # noqa: E402


def main():
    records = P.scan()
    print(f"{len(records)} images")
    X = P.thumbnails(records)
    n = len(records)
    sq = (X * X).sum(axis=1)

    # nearest neighbour for every image
    nearest = np.full(n, np.inf, dtype=np.float32)
    block = 2048
    for start in range(0, n, block):
        stop = min(start + block, n)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            gram = X[start:stop] @ X.T
        d2 = sq[start:stop, None] + sq[None, :] - 2.0 * gram
        rmse = np.sqrt(np.maximum(d2, 0.0) / X.shape[1])
        for local in range(stop - start):
            rmse[local, start + local] = np.inf
        nearest[start:stop] = rmse.min(axis=1)

    rng = np.random.default_rng(0)
    a, b = rng.integers(0, n, 200_000), rng.integers(0, n, 200_000)
    keep = a != b
    unrelated = np.sqrt(((X[a[keep]] - X[b[keep]]) ** 2).mean(axis=1))

    print("\nnearest-neighbour RMSE (what we would call a duplicate):")
    for q in (1, 5, 10, 25, 50, 75, 90):
        print(f"   p{q:<3} {np.percentile(nearest, q):7.2f}")

    print("\nunrelated-pair RMSE (what must NOT be caught):")
    for q in (0.01, 0.1, 0.5, 1, 5, 50):
        print(f"   p{q:<5} {np.percentile(unrelated, q):7.2f}")

    print(f"\n{'thresh':>7} {'imgs flagged':>13} {'% of set':>9} "
          f"{'unrelated pairs below':>22}")
    for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        flagged = int((nearest < t).sum())
        fp = float((unrelated < t).mean())
        print(f"{t:7.1f} {flagged:13d} {100 * flagged / n:8.1f}% "
              f"{100 * fp:21.3f}%")

    print("\nA usable threshold flags a real fraction of images while the "
          "unrelated-pair rate stays near zero.")


if __name__ == "__main__":
    main()
