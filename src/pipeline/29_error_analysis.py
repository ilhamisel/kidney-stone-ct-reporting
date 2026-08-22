# -*- coding: utf-8 -*-
"""Stage 29 — ERROR ANALYSIS.

Stages 24 and 26 report what the results are; this stage extracts the STRUCTURE of
the errors. There is no training: everything is computed from the held-out
prediction dumps that are already on disk.

Three panels and four tables:
  (a) has_stone sensitivity stratified by the reported diameter of the stone — a
      direct measurement of the partial-volume effect on the detection task, which
      elsewhere is only inferred from the size task. Kidneys misclassified in at
      least 80% of the 25 runs are also listed, as a queue for qualitative review.
  (b) The four-class laterality confusion matrix. A single accuracy figure does not
      show which direction the errors go, and the direction turns out to matter.
  (c) The triplet error taxonomy — which field the composite F1 breaks on. The
      reference triplets are partitioned EXACTLY into the categories.

Outputs:
  10_examples/figures/fig11_error_analysis.png
  11_paper/error_analysis_{model}.csv
  11_paper/error_analysis_persistent_{model}.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common.factset import canonical_stone_tuples
from common.ml import data as D
from common.ml import viz as V
from common.ml.runio import PREDS
from common.paths import LABELS, ROOT

_s = importlib.util.spec_from_file_location(
    "ev", str(Path(__file__).resolve().parent / "26_evaluate_reports.py"))
EV = importlib.util.module_from_spec(_s)
_s.loader.exec_module(EV)

OUT_FIG = ROOT / "10_examples" / "figures"
OUT_TAB = ROOT / "11_paper"
SEEDS = list(D.SEEDS)
LAT = ("NONE", "RIGHT", "LEFT", "BILATERAL")
# Referans tarafinda 'NONE' YOKTUR (197/197 vakada tas var), bu yuzden matris
# 3 referans satiri x 4 kestirim sutunudur; 'none' yalnizca kestirilebilir.
LAT_REF = ("RIGHT", "LEFT", "BILATERAL")

# Buckets by reported diameter. The bounds are the same half-open intervals as in
# common.labels.size_class, but the size3 merge ({<=3, 4-5} -> SMALL and
# {11-19, >=20} -> LARGE) is deliberately NOT applied: the gradient between those
# buckets is precisely what this panel exists to show.
BUCKETS = [("≤3 mm", 0, 4), ("4–5 mm", 4, 6), ("6–10 mm", 6, 11),
           ("11–19 mm", 11, 20), ("≥20 mm", 20, 1e9)]
# Triplet error taxonomy: an EXACT partition of the reference triplets.
CATS = ["exact match", "zone error only", "size error only", "zone + size error",
        "missed kidney", "detected, not characterized"]
CAT_COLORS = [V.GOOD, V.SEQ[3], V.SEQ[1], V.SERIES[1], V.CRITICAL, V.MUTED]


def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)


def bucket_of(mm) -> str | None:
    if mm is None or (isinstance(mm, float) and np.isnan(mm)):
        return None
    for name, lo, hi in BUCKETS:
        if lo <= float(mm) < hi:
            return name
    return None


# ----------------------------------------------------- (a) detection errors
def detection_errors(model: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """has_stone sensitivity and specificity pooled over the 25 runs, plus the
    kidneys that fail persistently rather than by chance."""
    files = sorted(PREDS.glob(f"e9_hs_{model}_has_stone_s*_f*.csv"))
    assert files, f"no prediction dump found: e9_hs_{model}"
    d = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    d["pred"] = (d["prob"].astype(float) >= 0.5).astype(int)
    mm = df.set_index("kidney_id")["max_mm"]
    d["bucket"] = d["kidney_id"].map(mm).map(bucket_of)

    rows = []
    pos = d[d["y"] == 1]
    for name, _, _ in BUCKETS:
        g = pos[pos["bucket"] == name]
        if len(g):
            rows.append({"stratum": name, "n_kidneys": g["kidney_id"].nunique(),
                         "n_run": len(g), "rate": float((g["pred"] == 1).mean()),
                         "kind": "sensitivity"})
    g = pos[pos["bucket"].isna()]
    if len(g):
        rows.append({"stratum": "size not\nrecorded", "n_kidneys": g["kidney_id"].nunique(),
                     "n_run": len(g), "rate": float((g["pred"] == 1).mean()),
                     "kind": "sensitivity"})
    neg = d[d["y"] == 0]
    rows.append({"stratum": "no reported\nstone", "n_kidneys": neg["kidney_id"].nunique(),
                 "n_run": len(neg), "rate": float((neg["pred"] == 0).mean()),
                 "kind": "specificity"})
    tab = pd.DataFrame(rows)

    # Kidneys wrong in >=80% of runs: systematic rather than stochastic failures,
    # and a small enough set for a radiologist to review individually.
    per = d.groupby("kidney_id").agg(y=("y", "first"), n=("pred", "size"),
                                     correct=("pred", lambda s: 0))
    per["correct"] = d.groupby("kidney_id").apply(
        lambda g: float((g["y"] == g["pred"]).mean()), include_groups=False)
    bad = per[per["correct"] <= 0.2].copy()
    bad["max_mm"] = bad.index.map(mm)
    bad["failure"] = np.where(bad["y"] == 1, "missed stone", "false alarm")
    plaus = d.groupby("kidney_id")["mask_plausible"].first()
    bad["mask_plausible"] = bad.index.map(plaus)
    bad = bad.sort_values(["failure", "max_mm"])[
        ["failure", "y", "max_mm", "mask_plausible", "correct"]]

    # segmentasyon guveni kalici hatalarda zenginlesiyor mu? (Fisher, tek yonlu)
    n_flag_bad = int((~bad["mask_plausible"].astype(bool)).sum())
    n_flag_all = int((~plaus.astype(bool)).sum())
    fisher = stats.fisher_exact(
        [[n_flag_bad, len(bad) - n_flag_bad],
         [n_flag_all - n_flag_bad, len(plaus) - len(bad) - (n_flag_all - n_flag_bad)]],
        alternative="greater")

    summary = {"n_pooled": len(d), "acc": float((d["y"] == d["pred"]).mean()),
               "fn_rate": float(((d["y"] == 1) & (d["pred"] == 0)).mean()),
               "fp_rate": float(((d["y"] == 0) & (d["pred"] == 1)).mean()),
               "n_persistent": int(len(bad)),
               "n_persistent_missed": int((bad["y"] == 1).sum()),
               "n_persistent_falsealarm": int((bad["y"] == 0).sum()),
               "n_mask_flag_all": n_flag_all, "n_mask_flag_persistent": n_flag_bad,
               "mask_flag_or": float(fisher.statistic), "mask_flag_p": float(fisher.pvalue),
               "n_kidneys": int(len(plaus))}
    return tab, bad, summary


# ------------------------ (b) and (c): decomposition of report-level errors
def report_errors(model: str, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict]:
    preds = EV.load_preds(f"e9_hs_{model}", f"e9_size_{model}", f"e9_2stage_{model}")
    cases = sorted(df["case_id"].unique())
    docs = {c: json.loads((LABELS / "reports" / f"{c}.json").read_text(encoding="utf-8"))
            for c in cases}

    cm = np.zeros((4, 4))                       # laterality; rows are the reference
    per_seed = defaultdict(list)
    for seed in SEEDS:
        cat, n_unchar = Counter(), 0
        extra = Counter()                       # predictions with no reference triplet
        for case in cases:
            sides = {}
            for side in ("RIGHT", "LEFT"):
                p = preds.get((seed, f"{case}_{side[0]}"), {})
                present = p.get("p_stone", 0.0) > 0.5
                sides[side] = {"present": present,
                               "cls": p.get("size_cls") if present else None,
                               "zone": p.get("zone") if present else None}
            fs_p = EV.predicted_factset(case, sides)
            fs_r = EV.reference_factset(docs[case])
            for key in ("right", "left"):
                if fs_p["kidneys"][key]["present"] and not fs_p["kidneys"][key]["stones"]:
                    n_unchar += 1
            cm[LAT.index(fs_r["laterality"]), LAT.index(fs_p["laterality"])] += 1

            pred = canonical_stone_tuples(fs_p)
            ref = EV.reduce_tuples(canonical_stone_tuples(fs_r))
            cp, cr = Counter(pred), Counter(ref)
            inter = cp & cr
            cat["exact match"] += sum(inter.values())
            rem_p = list((cp - inter).elements())
            rem_r = list((cr - inter).elements())

            # Both the reduced reference and the prediction hold at most one
            # triplet per side, so matching by side is one-to-one.
            for side in ("RIGHT", "LEFT"):
                pp = [t for t in rem_p if t[0] == side]
                rr = [t for t in rem_r if t[0] == side]
                if pp and rr:
                    zbad, sbad = pp[0][1] != rr[0][1], pp[0][2] != rr[0][2]
                    cat["zone + size error" if (zbad and sbad) else
                        ("zone error only" if zbad else "size error only")] += 1
                elif rr:                        # reference triplet with no counterpart
                    key = side.lower()
                    cat["missed kidney" if not fs_p["kidneys"][key]["present"]
                        else "detected, not characterized"] += 1
                elif pp:                        # prediction with no counterpart
                    key = side.lower()
                    extra["false-positive kidney" if not fs_r["kidneys"][key]["present"]
                          else "stone not characterized in reference"] += 1

        n_ref = sum(cat[c] for c in CATS)
        for c in CATS:
            per_seed[c].append(cat[c] / n_ref)
        for c in ("false-positive kidney", "stone not characterized in reference"):
            per_seed[c].append(extra[c] / n_ref)
        per_seed["_n_ref"].append(n_ref)
        per_seed["_n_unchar"].append(n_unchar)

    rows = [{"category": c, "mean": ci(v)[0], "ci_lo": ci(v)[1], "ci_hi": ci(v)[2],
             "per_seed": " ".join(f"{x:.4f}" for x in v)}
            for c, v in per_seed.items() if not c.startswith("_")]
    tab = pd.DataFrame(rows)
    lat_acc = float(np.trace(cm) / cm.sum())
    assert cm[LAT.index("NONE")].sum() == 0, "referansta 'NONE' lateralite beklenmiyordu"
    sub = cm[[LAT.index(x) for x in LAT_REF]]
    return sub / sub.sum(1, keepdims=True), tab, {
        "lat_acc": lat_acc, "n_ref_triplets": float(np.mean(per_seed["_n_ref"])),
        "n_uncharacterized": float(np.mean(per_seed["_n_unchar"])), "cm_counts": cm}


# -------------------------------------------------------------------- figure
def figure(det: pd.DataFrame, cm: np.ndarray, tri: pd.DataFrame,
           det_sum: dict, rep_sum: dict, model: str) -> Path:
    """Iki satirli yerlesim: (a) tam genislik, altta (b) ve (c).

    Tek satirli 3 panel ~2.9:1 en-boy veriyordu; Word figuru 6.3 inc genislige
    olceklendigi icin yazi 2 inc yukseklige sikisiyordu. 1.2:1 sayfa dostudur.
    """
    V.apply_theme()
    fig = plt.figure(figsize=(9.2, 7.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], width_ratios=[0.86, 1.14],
                          hspace=0.44, wspace=0.26,
                          left=0.085, right=0.985, top=0.85, bottom=0.07)

    # ---------------------------------------------- (a) tespit, capa gore
    ax = fig.add_subplot(gs[0, :])
    sens = det[det["kind"] == "sensitivity"]
    x = np.arange(len(det))
    colors = [V.SEQ[2] if k == "sensitivity" else V.MUTED for k in det["kind"]]
    ax.bar(x, det["rate"], color=colors, width=0.6, zorder=2)
    for xi, (_, r) in zip(x, det.iterrows()):
        ax.text(xi, r["rate"] + 0.022, f"{r['rate']:.3f}", ha="center", fontsize=8.4,
                color=V.INK, weight="semibold")
        ax.text(xi, 0.04, f"n={int(r['n_kidneys'])}", ha="center", fontsize=7.6,
                color=V.SURFACE if r["rate"] > 0.15 else V.MUTED)
    ax.axvline(len(sens) - 0.5, color=V.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.set_xticks(x, det["stratum"], fontsize=8.2)
    ax.set_ylim(0, 1.13), ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_ylabel("rate", fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_title("(a)  stone-presence detection, stratified by reported stone diameter",
                 pad=20)
    ax.text(0.0, 1.015, "sensitivity on stone-positive kidneys  ·  right of the dashed "
            "rule: specificity on kidneys with no reported stone",
            transform=ax.transAxes, fontsize=8, color=V.INK2, va="bottom")

    # ---------------------------------------------- (b) lateralite matrisi
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(cm, cmap=MF_cmap(), vmin=0, vmax=1, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0.002:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8.6,
                        weight="semibold" if LAT_REF[i] == LAT[j] else "normal",
                        color="#ffffff" if cm[i, j] > 0.55 else V.INK)
    ax.set_xticks(range(4), [t.lower() for t in LAT], fontsize=8.2)
    ax.set_yticks(range(3), [t.lower() for t in LAT_REF], fontsize=8.2)
    ax.set_xlabel("predicted", fontsize=9), ax.set_ylabel("reported", fontsize=9)
    ax.grid(False)
    ax.set_title("(b)  laterality", pad=20)
    ax.text(0.0, 1.02, f"row-normalized · overall {rep_sum['lat_acc']:.3f}",
            transform=ax.transAxes, fontsize=8, color=V.INK2, va="bottom")

    # ---------------------------------------------- (c) ucLU hata taksonomisi
    ax = fig.add_subplot(gs[1, 1])
    t = tri.set_index("category")
    rows = [(c, col) for c, col in zip(CATS, CAT_COLORS) if float(t.loc[c, "mean"]) > 0.0005]
    y = np.arange(len(rows))[::-1]                 # ilk kategori en ustte
    for yi, (c, col) in zip(y, rows):
        v, lo, hi = (float(t.loc[c, k]) for k in ("mean", "ci_lo", "ci_hi"))
        ax.barh([yi], [v], color=col, height=0.62, zorder=2)
        ax.hlines(yi, lo, hi, color=V.INK, lw=1.0, zorder=3)
        ax.vlines([lo, hi], yi - 0.11, yi + 0.11, color=V.INK, lw=1.0, zorder=3)
        ax.text(hi + 0.018, yi, f"{v:.3f}", va="center", fontsize=8.4,
                color=V.INK, weight="semibold")
    ax.set_yticks(y, [c for c, _ in rows], fontsize=8.2)
    ax.set_xlim(0, 0.68), ax.set_ylim(-0.62, len(rows) - 0.38)
    ax.set_xlabel("share of the reference triplets", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_title("(c)  composite triplet error taxonomy", pad=20)
    ax.text(0.0, 1.02, f"the modes partition the {rep_sum['n_ref_triplets']:.0f} reference "
            f"triplets exactly  ·  whiskers: 95% CI over 5 seeds",
            transform=ax.transAxes, fontsize=8, color=V.INK2, va="bottom")

    name = {"sst": "hybrid CNN-Transformer", "resnet18": "ResNet-18"}[model]
    fig.text(0.006, 0.972, f"Error analysis — {name}", ha="left", fontsize=13,
             weight="semibold", color=V.INK)
    fig.text(0.006, 0.947, f"(a) pooled over 25 runs (5 seeds × 5 folds); (b) and (c) "
             f"mean over 5 seeds, each a full pass over the cohort", ha="left",
             fontsize=8.6, color=V.INK2)
    fig.text(0.006, 0.925, f"kidney-level accuracy {det_sum['acc']:.3f}  ·  false negatives "
             f"{det_sum['fn_rate']:.1%}  ·  false positives {det_sum['fp_rate']:.1%}  ·  "
             f"{rep_sum['n_uncharacterized']:.0f} kidneys per seed were called "
             f"stone-positive but yield no scoreable triplet",
             ha="left", fontsize=8.6, color=V.MUTED)
    # the primary model takes the canonical filename; the comparison arm gets its own
    out = OUT_FIG / ("fig11_error_analysis.png" if model == "sst"
                     else f"fig11_error_analysis_{model}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def MF_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", [V.SURFACE] + V.SEQ)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sst", choices=["sst", "resnet18"])
    args = ap.parse_args()
    m = args.model

    df = D.load_table()
    det, bad, det_sum = detection_errors(m, df)
    cm, tri, rep_sum = report_errors(m, df)

    OUT_TAB.mkdir(exist_ok=True)
    det.to_csv(OUT_TAB / f"error_analysis_detection_{m}.csv", index=False,
               encoding="utf-8-sig")
    tri.to_csv(OUT_TAB / f"error_analysis_triplets_{m}.csv", index=False,
               encoding="utf-8-sig")
    bad.to_csv(OUT_TAB / f"error_analysis_persistent_{m}.csv", encoding="utf-8-sig")
    pd.DataFrame(rep_sum["cm_counts"], index=LAT, columns=LAT).to_csv(
        OUT_TAB / f"error_analysis_laterality_{m}.csv", encoding="utf-8-sig")
    out = figure(det, cm, tri, det_sum, rep_sum, m)

    print(f"model: {m}   havuz n={det_sum['n_pooled']}  acc={det_sum['acc']:.4f}  "
          f"FN={det_sum['fn_rate']:.4f}  FP={det_sum['fp_rate']:.4f}")
    print("\n(a) detection, by reported diameter")
    for _, r in det.iterrows():
        print(f"    {r['stratum'].replace(chr(10), ' '):<20} {r['kind'][:4]}  "
              f"{r['rate']:.3f}   (n={int(r['n_kidneys'])} kidneys)")
    print(f"\n    persistent errors (>=80% of runs): {det_sum['n_persistent']} kidneys "
          f"({det_sum['n_persistent_missed']} missed, "
          f"{det_sum['n_persistent_falsealarm']} false alarms)")
    print(f"\n(b) laterality accuracy {rep_sum['lat_acc']:.4f}")
    print(pd.DataFrame(np.round(cm, 3), index=LAT_REF, columns=LAT).to_string())
    print(f"    called stone-positive but not characterizable, per seed: "
          f"{rep_sum['n_uncharacterized']:.0f}")
    print(f"\n(c) triplet taxonomy (share of reference triplets, mean over seeds; "
          f"n_ref={rep_sum['n_ref_triplets']:.0f})")
    for _, r in tri.iterrows():
        print(f"    {r['category']:<42} {r['mean']:.4f} "
              f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
    print(f"\nwrote: {out.name} ({out.stat().st_size // 1024} KB) + 4 CSV files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
