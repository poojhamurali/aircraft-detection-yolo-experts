"""Aggregate the IR improvement study into tables, a verdict per idea, and figures.

Every number comes from the cached MPS predictions scored by improve_eval.score, so
all variants share one scorer. Choices that could be tuned (sequence post-processing
parameters, the deployment confidence threshold, which pipeline is "best") are made
on val only; test is read once with the chosen settings.

An idea is reported as an improvement only if, on test, it beats the baseline for
every seed (paired by seed) and its mean gain exceeds twice the baseline's seed-to-
seed standard deviation. Anything else is reported as within noise.

Usage:
    python scripts/summarize_improvements.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from improve_eval import (CLASS_NAMES, HERE, OUT, load_gts, load_preds, operating_point,
                          pred_path, score, temporal)

SEEDS = (0, 1, 2)
FIG = OUT / "figures"
VARIANTS = [  # key, label, run prefix, inference tag
    ("base640", "Baseline (640 px)", "imp_base640", "plain640"),
    ("tiled640", "Tiled inference (640 model)", "imp_base640", "tiled640"),
    ("tta960", "Test-time 960 px (640 model)", "imp_base640", "plain960"),
    ("img960", "Trained at 960 px", "imp_img960", "plain960"),
    ("img1280", "Trained at 1280 px", "imp_img1280", "plain1280"),
    ("hard640", "Hard-sample rebalancing", "imp_hard640", "plain640"),
    ("v8s640", "YOLOv8s baseline (640 px)", "imp_v8s640", "plain640"),
]
RUNS = HERE / "outputs" / "runs"
WEIGHTS = OUT / "weights"
TEMPORAL_GRID = [
    dict(relabel_share=r, prior=q, beta=b, causal=c)
    for c in (False, True)
    for r, q, b in itertools.product((None, 0.6, 0.75), (1.0, 0.5, 0.25), (1.0, 0.75, 0.5))
    if not (r is None and q == 1.0 and b == 1.0)
]


def available(prefix, tag):
    return [s for s in SEEDS if pred_path(f"{prefix}_s{s}", tag, "test").exists()
            and pred_path(f"{prefix}_s{s}", tag, "val").exists()]


def mean_std(xs):
    xs = np.asarray(xs, float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    gts = {s: load_gts(s) for s in ("val", "test")}
    preds = {}  # (key, seed, split) -> preds
    for key, _, prefix, tag in VARIANTS:
        for s in available(prefix, tag):
            for split in ("val", "test"):
                preds[(key, s, split)] = load_preds(f"{prefix}_s{s}", tag, split)

    def scores(key, split, post=None):
        out = {}
        for (k, s, sp), p in preds.items():
            if k == key and sp == split:
                out[s] = score(temporal(p, **post) if post else p, gts[split])
        return out

    # ---- 1. per-variant scores ------------------------------------------
    table = {}
    for key, label, _, _ in VARIANTS:
        per = {sp: scores(key, sp) for sp in ("val", "test")}
        if not per["test"]:
            continue
        row = {"label": label, "seeds": sorted(per["test"]), "per_seed": per}
        for sp in ("val", "test"):
            for m in ("mAP50", "mAP50_95", "precision", "recall"):
                row[f"{sp}_{m}"] = mean_std([v[m] for v in per[sp].values()])
            for c in ("airplane", "bird"):
                row[f"{sp}_{c}_mAP50"] = mean_std(
                    [v["per_class"].get(c, {}).get("mAP50", 0.0) for v in per[sp].values()])
        table[key] = row

    # ---- 2. sequence post-processing, tuned on baseline val --------------
    def val_mean(post):
        return np.mean([v["mAP50_95"] for v in scores("base640", "val", post).values()])

    grid = [(cfg, val_mean(cfg)) for cfg in TEMPORAL_GRID]
    best_off = max((g for g in grid if not g[0]["causal"]), key=lambda g: g[1])
    best_on = max((g for g in grid if g[0]["causal"]), key=lambda g: g[1])
    temporal_rows = {}
    for name, (cfg, _) in (("offline", best_off), ("causal", best_on)):
        for key in table:
            per = {sp: scores(key, sp, cfg) for sp in ("val", "test")}
            r = {"label": f"{table[key]['label']} + sequence ({name})", "config": cfg,
                 "per_seed": per, "seeds": sorted(per["test"])}
            for sp in ("val", "test"):
                for m in ("mAP50", "mAP50_95", "precision", "recall"):
                    r[f"{sp}_{m}"] = mean_std([v[m] for v in per[sp].values()])
                for c in ("airplane", "bird"):
                    r[f"{sp}_{c}_mAP50"] = mean_std(
                        [v["per_class"].get(c, {}).get("mAP50", 0.0) for v in per[sp].values()])
            temporal_rows[f"{key}+seq_{name}"] = r

    # ---- 3. verdicts vs baseline, paired by seed -------------------------
    base = table["base640"]
    noise = base["test_mAP50_95"][1]
    everything = {**table, **temporal_rows}

    def verdict(row, metric="mAP50_95"):
        common = [s for s in row["seeds"] if s in base["seeds"]]
        if not common:
            return None
        d = [row["per_seed"]["test"][s][metric] - base["per_seed"]["test"][s][metric]
             for s in common]
        md = float(np.mean(d))
        n_noise = base[f"test_{metric}"][1]
        real = len(common) > 1 and (all(x > 0 for x in d) or all(x < 0 for x in d)) \
            and abs(md) > 2 * n_noise
        return {"metric": metric, "paired_seeds": common, "deltas": d, "mean_delta": md,
                "noise_2sd": 2 * n_noise,
                "call": ("improvement" if md > 0 else "regression") if real else
                ("single seed - not testable" if len(common) < 2 else "within noise")}

    for k, r in everything.items():
        if k != "base640":
            r["verdict"] = {m: verdict(r, m) for m in ("mAP50_95", "mAP50")}

    # ---- 4. best pipeline by val, then confidence sweep ------------------
    best_key = max(everything, key=lambda k: everything[k]["val_mAP50_95"][0])
    thrs = np.round(np.arange(0.05, 0.951, 0.05), 3)

    def sweep(key):
        r = everything[key]
        variant = key.split("+")[0]
        cfg = r.get("config")
        curves = {}
        for sp in ("val", "test"):
            per_seed = []
            for s in r["seeds"]:
                p = preds[(variant, s, sp)]
                p = temporal(p, **cfg) if cfg else p
                per_seed.append([operating_point(p, gts[sp], t) for t in thrs])
            curves[sp] = [{m: float(np.mean([ps[i][m] for ps in per_seed]))
                           for m in ("precision", "recall", "f1")} | {"thr": float(t)}
                          for i, t in enumerate(thrs)]
        pick = max(range(len(thrs)), key=lambda i: curves["val"][i]["f1"])
        default = int(np.argmin(np.abs(thrs - 0.25)))
        return {"key": key, "curves": curves, "val_best_thr": float(thrs[pick]),
                "test_at_default_0.25": curves["test"][default],
                "test_at_val_best": curves["test"][pick]}

    sweeps = {"base640": sweep("base640")}
    if best_key != "base640":
        sweeps[best_key] = sweep(best_key)

    summary = {
        "protocol": {"model": "YOLO11s", "epochs": 15, "split": "datasets/clean",
                     "device": "mps", "seeds": list(SEEDS),
                     "baseline_test_mAP50_95_seed_sd": noise,
                     "improvement_rule": "better on every paired seed and mean gain > 2 x "
                                         "baseline seed SD (test)"},
        "variants": table, "sequence": temporal_rows,
        "sequence_grid_val": [{**c, "val_mAP50_95": float(v)} for c, v in grid],
        "sequence_best": {"offline": best_off[0], "causal": best_on[0]},
        "best_by_val": best_key, "confidence_sweeps": sweeps,
    }
    summary["weights"] = export_weights(table, everything, best_key)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_markdown(summary, everything)
    make_figures(summary, everything)
    print((OUT / "summary.md").read_text())


def export_weights(table, everything, best_key):
    """Copy every run's best.pt into one folder and name the val-selected winners.

    The overall winner is the variant chosen by val mAP50-95 (sequence post-processing,
    if part of it, is applied at inference and needs no weights), taking that variant's
    seed with the highest val score. The best YOLOv8s seed is exported alongside it.
    """
    import shutil
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    prefixes = {k: p for k, _, p, _ in VARIANTS}
    tags = {k: t for k, _, _, t in VARIANTS}
    for key, row in table.items():
        for s in row["seeds"]:
            src = RUNS / f"{prefixes[key]}_s{s}" / "weights" / "best.pt"
            if src.exists():
                shutil.copy2(src, WEIGHTS / f"{prefixes[key]}_s{s}.pt")

    def pick(key):
        row = table[key]
        s = max(row["seeds"], key=lambda x: row["per_seed"]["val"][x]["mAP50_95"])
        return s, row["per_seed"]["val"][s], row["per_seed"]["test"][s]

    chosen = {}
    winner = best_key.split("+")[0]
    for tag, key in (("BEST_IR_overall", winner), ("BEST_IR_yolov8s", "v8s640")):
        if key not in table:
            continue
        s, val, test = pick(key)
        name = f"{tag}__{prefixes[key]}_s{s}.pt"
        shutil.copy2(RUNS / f"{prefixes[key]}_s{s}" / "weights" / "best.pt", WEIGHTS / name)
        chosen[tag] = {"file": str(WEIGHTS / name), "variant": table[key]["label"],
                       "seed": s, "inference": tags[key],
                       "post_processing": everything[best_key].get("config")
                       if tag == "BEST_IR_overall" else None,
                       "val_mAP50_95": val["mAP50_95"], "test_mAP50": test["mAP50"],
                       "test_mAP50_95": test["mAP50_95"]}
    (WEIGHTS / "README.json").write_text(json.dumps(chosen, indent=2, default=float))

    downloads = Path.home() / "Downloads" / "IR_best_weights"
    try:
        downloads.mkdir(parents=True, exist_ok=True)
        for f in WEIGHTS.glob("BEST_IR_*.pt"):
            shutil.copy2(f, downloads / f.name)
        shutil.copy2(WEIGHTS / "README.json", downloads / "README.json")
    except OSError as e:
        print(f"could not copy winners to {downloads}: {e}")
    return chosen


def fmt(ms, nd=3):
    m, s = ms
    return f"{m:.{nd}f} ± {s:.{nd}f}" if s else f"{m:.{nd}f}"


def write_markdown(summary, everything):
    L = ["# IR improvement study (YOLO11s, 15 epochs, clean split, MPS)", "",
         f"Baseline seed-to-seed SD of test mAP@0.5:0.95: "
         f"{summary['protocol']['baseline_test_mAP50_95_seed_sd']:.4f}. "
         "Rule: " + summary["protocol"]["improvement_rule"] + ".", "",
         "| Variant | seeds | val mAP50-95 | test mAP50 | test mAP50-95 | airplane AP50 | "
         "bird AP50 | verdict (mAP50-95) |",
         "|---|---|---|---|---|---|---|---|"]
    for k, r in everything.items():
        v = r.get("verdict", {}).get("mAP50_95")
        call = "reference" if k == "base640" else (
            f"{v['call']} ({v['mean_delta']:+.3f})" if v else "-")
        L.append(f"| {r['label']} | {len(r['seeds'])} | {fmt(r['val_mAP50_95'])} | "
                 f"{fmt(r['test_mAP50'])} | {fmt(r['test_mAP50_95'])} | "
                 f"{fmt(r['test_airplane_mAP50'])} | {fmt(r['test_bird_mAP50'])} | {call} |")
    L += ["", f"Sequence settings chosen on val: offline {summary['sequence_best']['offline']}, "
          f"causal {summary['sequence_best']['causal']}.",
          f"Best pipeline by val mAP50-95: **{everything[summary['best_by_val']]['label']}**.", "",
          "## Confidence threshold (deployment operating point, micro P/R at IoU 0.5)", ""]
    for k, sw in summary["confidence_sweeps"].items():
        d, b = sw["test_at_default_0.25"], sw["test_at_val_best"]
        L.append(f"- {everything[k]['label']}: conf 0.25 -> P {d['precision']:.3f} "
                 f"R {d['recall']:.3f} F1 {d['f1']:.3f}; val-chosen conf {sw['val_best_thr']:.2f} "
                 f"-> P {b['precision']:.3f} R {b['recall']:.3f} F1 {b['f1']:.3f}")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


def make_figures(summary, everything):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(everything)
    labels = [everything[k]["label"].replace(" + sequence", "\n+ seq") for k in keys]
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(keys))
    for off, m, col in ((-0.2, "test_mAP50", "#4472C4"), (0.2, "test_mAP50_95", "#ED7D31")):
        mu = [everything[k][m][0] for k in keys]
        sd = [everything[k][m][1] for k in keys]
        ax.bar(x + off, mu, 0.4, yerr=sd, capsize=3, color=col,
               label=m.replace("test_", "test ").replace("_95", ":0.95").replace("mAP50", "mAP@0.5"))
    base = everything["base640"]
    for m, col in (("test_mAP50", "#4472C4"), ("test_mAP50_95", "#ED7D31")):
        ax.axhline(base[m][0], color=col, ls="--", lw=0.8)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("score (mean ± SD over seeds)")
    ax.set_title("IR improvement study - test set, YOLO11s 15 epochs (dashed = baseline)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "variants_test.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(summary["confidence_sweeps"]), figsize=(6.5 * len(
        summary["confidence_sweeps"]), 4.2), squeeze=False)
    for ax, (k, sw) in zip(axes[0], summary["confidence_sweeps"].items()):
        t = [c["thr"] for c in sw["curves"]["test"]]
        for m, col in (("precision", "#2E7D32"), ("recall", "#C62828"), ("f1", "#1F3864")):
            ax.plot(t, [c[m] for c in sw["curves"]["test"]], color=col, label=f"test {m}")
            ax.plot(t, [c[m] for c in sw["curves"]["val"]], color=col, ls=":", lw=1)
        ax.axvline(0.25, color="grey", ls="--", lw=0.8)
        ax.axvline(sw["val_best_thr"], color="black", lw=0.8)
        ax.set_title(f"{everything[k]['label']}\n(solid test, dotted val; black = val-best F1)",
                     fontsize=9)
        ax.set_xlabel("confidence threshold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "confidence_sweep.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
