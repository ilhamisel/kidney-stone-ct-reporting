# -*- coding: utf-8 -*-
"""Stage 26 — REPORT-LEVEL evaluation, the primary metric of this study.

The primary measure is agreement on structured fields: micro-averaged precision,
recall and F1 over `(side, zone, size_class)` triplets, together with explicit
hallucination and omission rates. Classification metrics alone are not enough,
because a report statement is correct only when every field of it is correct.

There is NO training here. All predictions are read from the held-out dumps under
the results directory. Each seed is one complete pass over the cohort — every
kidney appears in a test fold exactly once per seed — so the unit of variability
is the seed, and results are reported as the mean over 5 seeds with a 95% CI.

The predicted fact set is restricted to what the model can actually express: AT
MOST ONE stone per kidney, since the heads predict a maximum size class and a
dominant zone. Triplet metrics are therefore reported against two references:
  complete   every reported stone. Omission is partly structural here, because a
             multi-stone kidney cannot be represented at all.
  reduced    one triplet per kidney — the ceiling of the current output structure.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd
from scipy import stats

from common.factset import canonical_stone_tuples
from common.ml import data as D
from common.ml.runio import PREDS, RESULTS
from common.report_parser import parse_report
from common.templates import choose_variants, render_en, render_tr
from common.paths import LABELS, ROOT

SEEDS = list(D.SEEDS)
ZONES = ("LOWER", "MID", "UPPER")
SIZE3_TO_CLS = {0: "SMALL", 1: "MEDIUM", 2: "LARGE"}
# The model predicts three levels, so the five-class reference label is reduced to
# the SAME scale. Otherwise the two extreme classes could never match, and the
# comparison would penalize the model for an output space it was never given.
# Getting this wrong made the triplet F1 read 0.442 instead of 0.533.
TO_SIZE3 = {"MIKROLITIYAZIS": "SMALL", "KUCUK": "SMALL", "ORTA": "MEDIUM",
            "BUYUK": "LARGE", "COK_BUYUK": "LARGE"}
# The template works with the five-class keys, so each of the three levels is
# rendered by a representative class. This affects the wording only, never a fact.
SIZE3_TO_TEMPLATE = {"SMALL": "KUCUK", "MEDIUM": "ORTA", "LARGE": "BUYUK"}
OUT = ROOT / "11_paper"


# ------------------------------------------------------------------ kestirimler
def _seed_of(path: Path) -> int:
    """Seed from the _s<seed>_f<fold> suffix at the END of the filename.

    Splitting on '_s' anywhere would be unreliable, because configuration names
    contain it too (for example 'sst').
    """
    import re
    m = re.search(r"_s(\d+)_f(\d+)$", path.stem)
    assert m, f"unexpected filename: {path.name}"
    return int(m.group(1))


def load_preds(hs_cfg: str, size_cfg: str, zone_cfg: str) -> dict:
    """(seed, kidney_id) -> {p_stone, size_cls, zone}. Zone thresholds come from
    the run table, since they were fitted per fold."""
    runs = pd.read_csv(RESULTS / "runs.csv")
    out: dict[tuple[int, str], dict] = defaultdict(dict)

    for f in PREDS.glob(f"{hs_cfg}_has_stone_s*_f*.csv"):
        seed = _seed_of(f)
        for _, r in pd.read_csv(f).iterrows():
            v = r["prob"]
            out[(seed, r["kidney_id"])]["p_stone"] = float(
                json.loads(v) if isinstance(v, str) else v)

    for f in PREDS.glob(f"{size_cfg}_size3_s*_f*.csv"):
        seed = _seed_of(f)
        for _, r in pd.read_csv(f).iterrows():
            out[(seed, r["kidney_id"])]["size_cls"] = SIZE3_TO_CLS[
                int(np.argmax(json.loads(r["prob"])))]

    for f in PREDS.glob(f"{zone_cfg}_zone_s*_f*.csv"):
        seed = _seed_of(f)
        g = runs[(runs["config"] == zone_cfg) & (runs["seed"] == seed)]
        t1, t2 = float(g["thr_lo"].mean()), float(g["thr_hi"].mean())
        for _, r in pd.read_csv(f).iterrows():
            z = float(r["z_hat"])
            out[(seed, r["kidney_id"])]["zone"] = ZONES[(z >= t1) + (z >= t2)]
    return out


def predicted_factset(case_id: str, sides: dict) -> dict:
    """Build the canonical fact set from the model's predictions.

    The model predicts no diameter and no stone count, so mm and total_n are None
    rather than guessed. A generated report states only what was predicted.
    """
    kid = {}
    for key in ("right", "left"):
        p = sides[key.upper()]
        stones = ([{"zone": p["zone"], "mm": None, "cls": p["cls"]}]
                  if p["present"] and p["zone"] and p["cls"] else [])
        kid[key] = {"present": p["present"], "n_characterized": len(stones),
                    "stones": stones, "max_mm": None,
                    "max_cls": stones[0]["cls"] if stones else None,
                    "zones": sorted({s["zone"] for s in stones})}
    pres = [s for s in ("RIGHT", "LEFT") if sides[s]["present"]]
    lat = "NONE" if not pres else ("BILATERAL" if len(pres) == 2 else pres[0])
    return {"laterality": lat, "total_n": None, "total_qualifier": "EXACT",
            "n_characterized": sum(k["n_characterized"] for k in kid.values()),
            "kidneys": kid, "largest": None, "anomalies": []}


def reference_factset(doc: dict) -> dict:
    """Reference facts from the report JSON, in the same fact-set schema."""
    kid = {}
    for key in ("right", "left"):
        k = doc["targets"]["kidneys"][key]
        stones = [{"zone": s["zone"], "mm": s["size_mm"],
                   "cls": TO_SIZE3.get(s["size_class"])}
                  for s in doc["labels"]["stones"] if s["side"] == key.upper()]
        kid[key] = {"present": bool(k["has_stone"]), "n_characterized": len(stones),
                    "stones": stones, "max_mm": k["max_size_mm"],
                    "max_cls": TO_SIZE3.get(k["max_size_class"]),
                    "zones": sorted({s["zone"] for s in stones})}
    pres = [s for s in ("RIGHT", "LEFT") if kid[s.lower()]["present"]]
    lat = "NONE" if not pres else ("BILATERAL" if len(pres) == 2 else pres[0])
    return {"laterality": lat, "kidneys": kid}


def reduce_tuples(tuples: list) -> list:
    """One triplet per kidney — the ceiling of the current output structure.

    Where a kidney holds several stones, the largest class and its zone are kept,
    which is the same reduction the model's own heads perform.
    """
    order = {"SMALL": 0, "MEDIUM": 1, "LARGE": 2}
    best: dict[str, tuple] = {}
    for t in tuples:
        side, _, cls = t
        if side not in best or order.get(cls, -1) > order.get(best[side][2], -1):
            best[side] = t
    return sorted(best.values())


def micro_counts(pred: list, ref: list) -> tuple[int, int, int]:
    """Multiset matching -> (true positives, false positives, false negatives)."""
    r = list(ref)
    tp = 0
    for t in pred:
        if t in r:
            r.remove(t)
            tp += 1
    return tp, len(pred) - tp, len(r)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2:
        return (v.mean() if len(v) else float("nan"),) * 3
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


def fmt(t):
    return f"{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]"


# ----------------------------------------------------------------- main flow
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sst", choices=["sst", "resnet18"])
    args = ap.parse_args()
    m = args.model
    preds = load_preds(f"e9_hs_{m}", f"e9_size_{m}", f"e9_2stage_{m}")

    df = D.load_table()
    cases = sorted(df["case_id"].unique())
    docs = {c: json.loads((LABELS / "reports" / f"{c}.json").read_text(encoding="utf-8"))
            for c in cases}

    acc = defaultdict(list)          # one entry per seed
    per_case_rows = []
    for seed in SEEDS:
        n_lat_ok = n_case = 0
        k_tp = k_fp = k_fn = 0                      # kidney-level has_stone
        cls_ok = cls_n = zone_ok = zone_n = 0
        full = [0, 0, 0]
        red = [0, 0, 0]
        rt_ok = rt_n = 0
        no_size_pred = 0

        for case in cases:
            doc = docs[case]
            sides = {}
            for side in ("RIGHT", "LEFT"):
                kid = f"{case}_{side[0]}"
                p = preds.get((seed, kid), {})
                present = p.get("p_stone", 0.0) > 0.5
                if present and "size_cls" not in p:
                    no_size_pred += 1
                sides[side] = {"present": present,
                               "cls": p.get("size_cls") if present else None,
                               "zone": p.get("zone") if present else None}

            fs_p, fs_r = predicted_factset(case, sides), reference_factset(doc)
            n_case += 1
            n_lat_ok += fs_p["laterality"] == fs_r["laterality"]

            for side in ("right", "left"):
                yp = fs_p["kidneys"][side]["present"]
                yr = fs_r["kidneys"][side]["present"]
                k_tp += yp and yr
                k_fp += yp and not yr
                k_fn += yr and not yp
                if yr and fs_r["kidneys"][side]["max_cls"]:
                    cls_n += 1
                    cls_ok += fs_p["kidneys"][side]["max_cls"] == fs_r["kidneys"][side]["max_cls"]
                zr = fs_r["kidneys"][side]["zones"]
                if yr and len(zr) == 1:
                    zone_n += 1
                    zone_ok += fs_p["kidneys"][side]["zones"] == zr

            tp_, ref_ = canonical_stone_tuples(fs_p), canonical_stone_tuples(fs_r)
            for tgt, ref_list in ((full, ref_), (red, reduce_tuples(ref_))):
                a, b, c = micro_counts(tp_, ref_list)
                tgt[0] += a
                tgt[1] += b
                tgt[2] += c

            # does the generated text add or drop a fact? the gate applied to the
            # model's own reports, so that report-level error can be attributed
            var = choose_variants(case, doc["reports"]["generator_seed"])
            fs_render = json.loads(json.dumps(fs_p))     # copy, rescaled for the template
            for key in ("right", "left"):
                for s in fs_render["kidneys"][key]["stones"]:
                    s["cls"] = SIZE3_TO_TEMPLATE[s["cls"]]
            for lang, render in (("tr", render_tr), ("en", render_en)):
                txt = render(fs_render, var)["full"]
                back = parse_report(txt, lang)
                back_t = [(a, b, TO_SIZE3.get(c, c)) for a, b, c in
                          canonical_stone_tuples(back)]
                rt_n += 1
                rt_ok += sorted(back_t) == sorted(tp_) and \
                    back["laterality"] == fs_p["laterality"]

            if seed == SEEDS[0]:
                per_case_rows.append({
                    "case_id": case, "lat_ref": fs_r["laterality"],
                    "lat_pred": fs_p["laterality"],
                    "tuples_ref": json.dumps(ref_), "tuples_pred": json.dumps(tp_)})

        acc["laterality_acc"].append(n_lat_ok / n_case)
        p_, r_, f_ = prf(k_tp, k_fp, k_fn)
        acc["kidney_p"].append(p_), acc["kidney_r"].append(r_), acc["kidney_f1"].append(f_)
        acc["size_cls_acc"].append(cls_ok / cls_n if cls_n else float("nan"))
        acc["zone_acc"].append(zone_ok / zone_n if zone_n else float("nan"))
        for name, tgt in (("full", full), ("red", red)):
            p_, r_, f_ = prf(*tgt)
            acc[f"{name}_p"].append(p_), acc[f"{name}_r"].append(r_), acc[f"{name}_f1"].append(f_)
            acc[f"{name}_halluc"].append(tgt[1] / (tgt[0] + tgt[1]) if tgt[0] + tgt[1] else 0.0)
            acc[f"{name}_omit"].append(tgt[2] / (tgt[0] + tgt[2]) if tgt[0] + tgt[2] else 0.0)
        acc["roundtrip"].append(rt_ok / rt_n)
        acc["no_size_pred"].append(no_size_pred)

    print(f"REPORT-LEVEL EVALUATION — model {m}, {len(cases)} cases x {len(SEEDS)} seeds\n")
    print("Field level (mean +/- 95% CI over seeds)")
    print(f"  laterality accuracy (4-class)  : {fmt(ci(acc['laterality_acc']))}")
    print(f"  kidney has_stone  P            : {fmt(ci(acc['kidney_p']))}")
    print(f"                    R            : {fmt(ci(acc['kidney_r']))}")
    print(f"                    F1           : {fmt(ci(acc['kidney_f1']))}")
    print(f"  max_size_class accuracy        : {fmt(ci(acc['size_cls_acc']))}   (stone-positive kidneys)")
    print(f"  zone accuracy (single-zone)    : {fmt(ci(acc['zone_acc']))}")
    print("\n(side, zone, size_class) triplets — micro-averaged")
    for name, lbl in (("red", "reduced reference (1 stone per kidney)"),
                      ("full", "complete reference (all reported stones)")):
        print(f"  {lbl}")
        print(f"     P {fmt(ci(acc[f'{name}_p']))}   R {fmt(ci(acc[f'{name}_r']))}"
              f"   F1 {fmt(ci(acc[f'{name}_f1']))}")
        print(f"     hallucination {fmt(ci(acc[f'{name}_halluc']))}"
              f"   atlama {fmt(ci(acc[f'{name}_omit']))}")
    print(f"\nRound-trip gate on the model's reports: {fmt(ci(acc['roundtrip']))}"
          f"  ({int(np.mean(acc['roundtrip']) * len(cases) * 2)}/{len(cases) * 2} metin, TR+EN)")
    print(f"Positive kidneys with no size prediction: median {np.median(acc['no_size_pred']):.0f}"
          f"/seed  (false positives; they yield no scoreable triplet)")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"metric": k, "mean": ci(v)[0], "ci_lo": ci(v)[1], "ci_hi": ci(v)[2],
             "per_seed": json.dumps([round(float(x), 4) for x in v])}
            for k, v in acc.items()]
    pd.DataFrame(rows).to_csv(OUT / f"report_metrics_{m}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_case_rows).to_csv(OUT / f"report_percase_{m}_s{SEEDS[0]}.csv",
                                       index=False, encoding="utf-8-sig")
    print(f"\nwrote -> {OUT / f'report_metrics_{m}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
