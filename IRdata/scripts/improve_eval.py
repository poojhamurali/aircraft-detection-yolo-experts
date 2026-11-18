"""Inference, scoring and post-processing for the IR improvement experiments.

Predictions are cached per (run, inference mode, split) as raw low-threshold
detections, and everything downstream (mAP, sequence post-processing, confidence
sweeps) is computed from that cache, so post-processing can be tuned on val and
re-scored on test without re-running the network.

mAP follows ultralytics exactly: the same greedy IoU matching at the ten thresholds
0.50:0.95 and `ultralytics.utils.metrics.ap_per_class`. `check` compares this scorer
with the `model.val()` numbers stored in a run's metrics.json.

Sequence post-processing. Frames of one video are on average 37 frames apart here, too far
apart to link boxes by position, so the evidence shared between frames is which
class a video contains and whether a detection is corroborated elsewhere in it.
A frame's video is taken from the `IR_<CLASS>_<nnn>` stem as an opaque group id; the
class word in that id is never read. Unnamed `IR_frame_<n>` images carry no video id
and are left untouched. Three operations, all optional:

  relabel   if one class holds at least `share` of the confidence mass of a
            video's detections (conf >= vote_conf), every detection of another class
            also gets a copy under the majority class at conf * share. The original
            hypothesis is kept, so a wrong vote costs a false alarm, not a hit;
  prior     conf <- conf * (prior + (1 - prior) * m_c / max m), where m is that class
            mass over the other frames of the video;
  support   conf <- conf * (beta + (1 - beta) * s), where s is the highest confidence
            of a same-class detection in another frame of the video. Isolated false
            alarms in a video that otherwise never shows that class are suppressed.

`causal=True` restricts all three to the current and earlier frames (online use);
otherwise the whole video is used (offline review of a recording).

Usage:
    python scripts/improve_eval.py predict --run imp_base640_s0 --imgsz 640 --mode plain
    python scripts/improve_eval.py check --run imp_base640_s0
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "datasets" / "clean"
OUT = HERE / "outputs" / "improvements"
PRED_DIR = OUT / "preds"
CLASS_NAMES = ["airplane", "bird", "drone", "helicopter"]
IOUV = np.linspace(0.5, 0.95, 10)
SEQ_RE = re.compile(r"^(IR_[A-Z]+_\d+)_frame_(\d+)_jpg")
TILE, TILE_STEP = 384, 256


# --------------------------------------------------------------------------
# inference
def _nms(boxes, scores, classes, iou):
    import torch
    from torchvision.ops import batched_nms
    if len(boxes) == 0:
        return np.zeros(0, int)
    keep = batched_nms(torch.as_tensor(boxes, dtype=torch.float32),
                       torch.as_tensor(scores, dtype=torch.float32),
                       torch.as_tensor(classes), iou)
    return keep.numpy()


def _boxes(res, dx=0.0, dy=0.0):
    b = res.boxes
    xyxy = b.xyxy.cpu().numpy().astype(float)
    xyxy[:, [0, 2]] += dx
    xyxy[:, [1, 3]] += dy
    return xyxy, b.conf.cpu().numpy().astype(float), b.cls.cpu().numpy().astype(int)


def predict_images(model, paths, imgsz, mode, device, batch=16):
    """Raw detections (conf >= 0.001) per image stem.

    plain  one pass at `imgsz`.
    tiled  the plain pass plus a 2x2 grid of 384 px overlapping crops, each resized to
           `imgsz` (a 1.67x zoom at 640); tile boxes touching an interior tile edge are
           dropped (the full-image pass covers those), then class-wise NMS merges all.
    """
    import cv2

    kw = dict(imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, device=device, verbose=False)
    out = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        for p, res in zip(chunk, model.predict([str(x) for x in chunk], **kw)):
            out[p.stem] = list(_boxes(res))
    if mode == "plain":
        return {k: _pack(*v, *_hw(paths[0])) for k, v in out.items()}

    for p in paths:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        crops, offs = [], []
        for y in range(0, max(1, h - TILE) + 1, TILE_STEP):
            for x in range(0, max(1, w - TILE) + 1, TILE_STEP):
                crops.append(img[y:y + TILE, x:x + TILE])
                offs.append((x, y))
        boxes, confs, clss = out[p.stem]
        boxes, confs, clss = [boxes], [confs], [clss]
        for (x, y), res in zip(offs, model.predict(crops, **kw)):
            b, c, k = _boxes(res, x, y)
            edge = np.zeros(len(b), bool)
            for side, lim, interior in ((0, x, x > 0), (1, y, y > 0),
                                        (2, x + TILE, x + TILE < w), (3, y + TILE, y + TILE < h)):
                if interior:
                    edge |= np.abs(b[:, side] - lim) < 2
            boxes.append(b[~edge]); confs.append(c[~edge]); clss.append(k[~edge])
        b, c, k = np.concatenate(boxes), np.concatenate(confs), np.concatenate(clss)
        keep = _nms(b, c, k, 0.6)[:300]
        out[p.stem] = [b[keep], c[keep], k[keep]]
    return {k: _pack(*v, *_hw(paths[0])) for k, v in out.items()}


def _hw(path):
    import cv2
    h, w = cv2.imread(str(path)).shape[:2]
    return h, w


def _pack(b, c, k, h, w):
    return {"boxes": np.round(b, 2).tolist(), "conf": np.round(c, 5).tolist(),
            "cls": k.tolist(), "hw": [h, w]}


def pred_path(run, tag, split):
    return PRED_DIR / f"{run}__{tag}__{split}.json"


def load_preds(run, tag, split):
    raw = json.loads(pred_path(run, tag, split).read_text())
    return {k: {"boxes": np.asarray(v["boxes"], float).reshape(-1, 4),
                "conf": np.asarray(v["conf"], float), "cls": np.asarray(v["cls"], int),
                "hw": v["hw"]} for k, v in raw.items()}


# --------------------------------------------------------------------------
# scoring
def load_gts(split):
    from confusion_analysis import load_gt
    gts = {}
    for img in sorted((DATA / "images" / split).glob("*.jpg")):
        b, c = load_gt(DATA / "labels" / split / f"{img.stem}.txt", 640, 640)
        gts[img.stem] = (np.asarray(b, float).reshape(-1, 4), np.asarray(c, int))
    return gts


def _iou(a, b):
    from confusion_analysis import iou_matrix
    return iou_matrix(a, b)


def _match(pred_cls, gt_cls, iou):
    """ultralytics BaseValidator.match_predictions (greedy), iou is pred x gt."""
    correct = np.zeros((len(pred_cls), len(IOUV)), bool)
    if len(pred_cls) == 0 or len(gt_cls) == 0:
        return correct
    iou = iou.T * (gt_cls[:, None] == pred_cls[None, :])  # gt x pred
    for i, t in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= t)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def score(preds, gts):
    from ultralytics.utils.metrics import ap_per_class
    tps, confs, pcls, tcls = [], [], [], []
    for stem, (gb, gc) in gts.items():
        p = preds.get(stem)
        tcls.append(gc)
        if p is None or len(p["cls"]) == 0:
            continue
        sx, sy = 640 / p["hw"][1], 640 / p["hw"][0]
        pb = p["boxes"] * [sx, sy, sx, sy]
        tps.append(_match(p["cls"], gc, _iou(pb, gb)))
        confs.append(p["conf"]); pcls.append(p["cls"])
    tp = np.concatenate(tps) if tps else np.zeros((0, 10), bool)
    out = ap_per_class(tp, np.concatenate(confs) if confs else np.zeros(0),
                       np.concatenate(pcls) if pcls else np.zeros(0),
                       np.concatenate(tcls))
    p, r, ap, classes = out[2], out[3], out[5], out[6]
    res = {"mAP50": float(ap[:, 0].mean()), "mAP50_95": float(ap.mean()),
           "precision": float(p.mean()), "recall": float(r.mean()), "per_class": {}}
    for i, c in enumerate(classes):
        res["per_class"][CLASS_NAMES[int(c)]] = {
            "precision": float(p[i]), "recall": float(r[i]),
            "mAP50": float(ap[i, 0]), "mAP50_95": float(ap[i].mean())}
    return res


def operating_point(preds, gts, thr, iou_t=0.5):
    """Micro precision / recall / F1 of detections with conf >= thr (greedy, IoU 0.5)."""
    tp = fp = n_gt = 0
    for stem, (gb, gc) in gts.items():
        n_gt += len(gc)
        p = preds.get(stem)
        if p is None:
            continue
        keep = p["conf"] >= thr
        if not keep.any():
            continue
        c = _match(p["cls"][keep], gc, _iou(p["boxes"][keep], gb))[:, 0]
        tp += int(c.sum()); fp += int((~c).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n_gt if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"thr": round(thr, 3), "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": n_gt - tp}


# --------------------------------------------------------------------------
# sequence post-processing
def _seq(stem):
    m = SEQ_RE.match(stem)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def temporal(preds, relabel_share=None, beta=1.0, prior=1.0, vote_conf=0.25, causal=False):
    seqs = defaultdict(list)
    for stem in preds:
        sid, fr = _seq(stem)
        if sid is not None:
            seqs[sid].append((fr, stem))
    out = {k: {**v, "boxes": v["boxes"].copy(), "conf": v["conf"].copy(),
               "cls": v["cls"].copy()} for k, v in preds.items()}

    def mass(stems):
        m = np.zeros(len(CLASS_NAMES))
        for s in stems:
            p = preds[s]
            k = p["conf"] >= vote_conf
            np.add.at(m, p["cls"][k], p["conf"][k])
        return m

    for frames in seqs.values():
        frames.sort()
        stems = [s for _, s in frames]
        for idx, stem in enumerate(stems):
            cur, orig = out[stem], preds[stem]
            others = [s for j, s in enumerate(stems) if j != idx and (not causal or j < idx)]
            if relabel_share is not None:
                m = mass(stems[:idx + 1] if causal else stems)
                if m.sum() > 0 and m.max() / m.sum() >= relabel_share:
                    major, share = int(m.argmax()), m.max() / m.sum()
                    alt = orig["cls"] != major
                    cur["boxes"] = np.concatenate([cur["boxes"], orig["boxes"][alt]])
                    cur["conf"] = np.concatenate([cur["conf"], orig["conf"][alt] * share])
                    cur["cls"] = np.concatenate([cur["cls"], np.full(alt.sum(), major)])
            if prior < 1.0 and others:
                m = mass(others)
                if m.sum() > 0:
                    rel = m / m.max()
                    cur["conf"] = cur["conf"] * (prior + (1 - prior) * rel[cur["cls"]])
            if beta < 1.0 and others:
                best = np.zeros(len(CLASS_NAMES))
                for s in others:
                    np.maximum.at(best, preds[s]["cls"], preds[s]["conf"])
                cur["conf"] = cur["conf"] * (beta + (1 - beta) * best[cur["cls"]])
    if relabel_share is not None:
        for stem, p in out.items():
            if len(p["cls"]):
                keep = _nms(p["boxes"], p["conf"], p["cls"], 0.7)[:300]
                out[stem] = {**p, "boxes": p["boxes"][keep], "conf": p["conf"][keep],
                             "cls": p["cls"][keep]}
    return out


# --------------------------------------------------------------------------
def cmd_predict(args):
    from ultralytics import YOLO
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(HERE / "outputs" / "runs" / args.run / "weights" / "best.pt"))
    tag = f"{args.mode}{args.imgsz}"
    for split in ("val", "test"):
        dst = pred_path(args.run, tag, split)
        if dst.exists() and not args.force:
            print(f"cached {dst.name}")
            continue
        paths = sorted((DATA / "images" / split).glob("*.jpg"))
        preds = predict_images(model, paths, args.imgsz, args.mode, args.device)
        dst.write_text(json.dumps(preds))
        m = score(load_preds(args.run, tag, split), load_gts(split))
        print(f"{args.run} {tag} {split}: mAP50 {m['mAP50']:.4f}  mAP50-95 {m['mAP50_95']:.4f}")


def cmd_check(args):
    ref = json.loads((HERE / "outputs" / "runs" / args.run / "metrics.json").read_text())
    for split in ("val", "test"):
        m = score(load_preds(args.run, "plain" + str(ref["imgsz"]), split), load_gts(split))
        print(f"{split}: ours {m['mAP50']:.4f}/{m['mAP50_95']:.4f}   ultralytics "
              f"{ref[split]['mAP50']:.4f}/{ref[split]['mAP50_95']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--run", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--mode", default="plain", choices=["plain", "tiled"])
    p.add_argument("--device", default="mps")
    p.add_argument("--force", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--run", required=True)
    args = ap.parse_args()
    {"predict": cmd_predict, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
