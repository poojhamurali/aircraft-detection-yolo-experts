"""Check the MPS workaround against the code it replaces.

Run on CPU, where the original boolean-mask indexing is correct, so any
disagreement is a bug in the replacement rather than in the backend.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics.utils.tal import TaskAlignedAssigner  # noqa: E402

import mps_tal_patch  # noqa: E402


def main():
    torch.manual_seed(0)
    b, n_max, na, nc = 4, 40, 8400, 1

    assigner = TaskAlignedAssigner(topk=10, num_classes=nc)
    assigner.bs, assigner.n_max_boxes = b, n_max

    pd_scores = torch.rand(b, na, nc)
    pd_bboxes = torch.rand(b, na, 4).cumsum(-1)
    gt_labels = torch.zeros(b, n_max, 1)
    gt_bboxes = torch.rand(b, n_max, 4).cumsum(-1)
    mask_gt = torch.rand(b, n_max, na) > 0.7

    args = (pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)
    old = TaskAlignedAssigner.get_box_metrics(assigner, *args)
    new = mps_tal_patch._get_box_metrics(assigner, *args)

    ok = True
    for name, o, n in zip(("align_metric", "overlaps"), old, new):
        same = torch.allclose(o, n, atol=1e-6)
        ok &= same and o.shape == n.shape
        print(f"{name:<13} shape {tuple(o.shape)} vs {tuple(n.shape)}  "
              f"max abs diff {(o - n).abs().max().item():.3e}  allclose={same}")

    print("mps available:", torch.backends.mps.is_available())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
