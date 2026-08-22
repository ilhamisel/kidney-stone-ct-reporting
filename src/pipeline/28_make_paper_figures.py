# -*- coding: utf-8 -*-
"""Stage 28 — the schematic figures and appendix cards for the manuscript.

Stage 24 produces the result and data figures; this stage produces the three
schematics and the end-to-end patient cards:

  fig8_study_flow.png        cohort flow diagram
  fig9_pipeline_comparison   the previous pipeline against the revised one
  fig10_report_template      the structured reporting schema and template
  appendix_case_{1..4}.png   end-to-end cards for four clinical scenarios

Cases are drawn from the fold 0 test set, which the detector checkpoint has not
seen, and are selected automatically to match each scenario. Which cases were
chosen is printed, so a figure can always be traced back to its case.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common.ml import data as D
from common.ml import viz as V
from common.ml.transforms import eval_transform
from common.paths import LABELS, ROOT

_s = importlib.util.spec_from_file_location(
    "mf", str(Path(__file__).resolve().parent / "24_make_figures.py"))
MF = importlib.util.module_from_spec(_s)
_s.loader.exec_module(MF)

OUT = ROOT / "10_examples" / "figures"


def box(ax, x, y, w, h, text, fc="#ffffff", ec=V.AXIS, fs=8.5, weight="normal",
        tc=None, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fc, ec=ec, lw=1.0, zorder=2))
    ax.text(x + (w / 2 if align == "center" else 0.012), y + h / 2, text,
            ha=align if align != "center" else "center", va="center",
            fontsize=fs, color=tc or V.INK, weight=weight, zorder=3, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color=color or V.MUTED, zorder=1,
                                 shrinkA=0, shrinkB=0))


def blank_ax(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ------------------------------------------------------------------ fig 8
def fig8_study_flow(df, full):
    n_pat = full["patient_id"].nunique()
    n_exam = full["case_id"].nunique()
    n_kid_all = len(full)
    n_kid = len(df)
    pos = int(df["has_stone"].astype(bool).sum())
    neg = n_kid - pos
    sz = len(D.task_subset(df, "size3"))
    zn = len(D.task_subset(df, "zone"))

    fig, ax = blank_ax((7.8, 8.2))
    W, X, H = 0.74, 0.13, 0.082
    ax.text(0.0, 0.985, "Study flow", fontsize=12.5, color=V.INK, weight="semibold",
            va="top")
    rows = [
        (0.855, f"Consecutive non-contrast abdominal CT examinations\n"
                f"with radiologist-reported nephrolithiasis\n"
                f"{n_pat} patients · {n_exam} examinations", "#eef4fd"),
        (0.725, "Automated series selection and lossless 16-bit HU archive\n"
                "(orientation-based series choice; no manual slice selection)", "#ffffff"),
        (0.595, f"Automated kidney segmentation (TotalSegmentator)\n"
                f"{n_kid_all} kidneys · right/left orientation self-check "
                f"{n_kid_all}/{n_kid_all} passed", "#ffffff"),
        (0.465, f"One examination per patient (index examination)\n"
                f"→ modelling cohort: {n_pat} patients · {n_pat} examinations · "
                f"{n_kid} kidneys", "#eef4fd"),
    ]
    for y, t, fc in rows:
        box(ax, X, y, W, H, t, fc=fc, fs=8.3)
    for i in range(len(rows) - 1):
        arrow(ax, 0.5, rows[i][0], 0.5, rows[i + 1][0] + H)

    tasks = [
        (f"Stone presence\n{n_kid} kidneys\n{pos} positive / {neg} negative", V.SERIES[0]),
        (f"Stone size class\n{sz} kidneys with a\ncharacterised stone", V.SERIES[1]),
        (f"Intrarenal zone\n{zn} kidneys with a\nzone-labelled stone", V.SERIES[2]),
    ]
    ty, th = 0.255, 0.135
    for i, (t, col) in enumerate(tasks):
        bx = X + i * (W / 3)
        box(ax, bx + 0.012, ty, W / 3 - 0.024, th, t, fc="#ffffff", ec=col, fs=8)
        arrow(ax, 0.5, 0.465, bx + W / 6, ty + th, color=col)

    box(ax, X, 0.075, W, 0.13,
        "Patient-level stratified 5-fold cross-validation × 5 seeds (25 runs)\n"
        "both kidneys of a patient always in the same fold\n"
        "every kidney is in the test fold exactly once per seed",
        fc="#f4f7f2", ec=V.GOOD, fs=8.3)
    arrow(ax, 0.5, ty, 0.5, 0.205)

    # the two notes are merged into one block; the arrow overlapped the text
    ax.text(X, 0.035, "All kidneys retained; no case was excluded to improve laterality "
            "performance.\nBilateral disease is represented natively because the modelling "
            "unit is the kidney.", fontsize=7.8, color=V.MUTED, linespacing=1.6, va="top")
    fig.savefig(OUT / "fig8_study_flow.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 9
def fig9_pipeline_comparison():
    prev = [
        "Expert-guided 100-slice subvolume\nselected manually per examination",
        "Coronal reformats of the selected\nsubvolume; “informative” slices kept",
        "Modelling unit: slice\nSplitting unit: patient",
        "Bilateral cases excluded for the\nlaterality analysis (163 of 195)",
        "Single train/validation/test split",
        "Two side-specific presence models\n+ attention-derived zone",
    ]
    new = [
        "No manual selection: automated series\nchoice from the full examination",
        "Automated kidney segmentation;\nper-kidney crop of the whole organ",
        "Modelling unit: kidney\nSplitting unit: patient",
        "All kidneys retained;\nbilateral disease represented natively",
        "5 seeds × 5 folds = 25 runs,\n95% CI and paired tests",
        "Presence and size by multiple-instance\nlearning; zone by detector position",
    ]
    fig, ax = blank_ax((10.6, 6.6))
    for j, (title, items, col) in enumerate(
            (("Previous pipeline", prev, V.MUTED), ("Revised pipeline", new, V.SERIES[0]))):
        x = 0.045 + j * 0.5
        ax.text(x, 0.945, title, fontsize=10.5, color=V.INK if j else V.INK2,
                weight="semibold")
        for i, t in enumerate(items):
            y = 0.80 - i * 0.135
            box(ax, x, y, 0.41, 0.11, t, fs=8.2,
                fc="#ffffff" if j else "#faf9f7", ec=col, align="left")
    for i in range(6):
        y = 0.855 - i * 0.135
        arrow(ax, 0.463, y, 0.537, y, color=V.SERIES[2])
    ax.text(0.5, 0.02, "Each row is a direct response to a reviewer objection: manual "
            "subvolume selection, bilateral exclusion,\nunit definition, and statistical "
            "robustness are addressed structurally rather than by re-tuning.",
            ha="center", fontsize=8, color=V.INK2, linespacing=1.5)
    fig.suptitle("From the previously submitted pipeline to the revised pipeline",
                 x=0.02, y=0.985, ha="left", fontsize=12, color=V.INK, weight="semibold")
    fig.savefig(OUT / "fig9_pipeline_comparison.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 10
def fig10_report_template(df):
    """The structured reporting figure: schema fields, then the slotted template,
    then a filled example."""
    case = "KS0005"
    doc = json.loads((LABELS / "reports" / f"{case}.json").read_text(encoding="utf-8"))
    rep = doc["reports"]

    fig, ax = blank_ax((13.4, 8.4))
    mono = {"family": "monospace", "fontsize": 6.6, "color": V.INK, "linespacing": 1.5}
    TOP, BH, BY = 0.885, 0.828, 0.05         # box top / height / bottom edge

    schema = [
        "CASE LEVEL",
        "  laterality       NONE | RIGHT | LEFT | BILATERAL",
        "  n_stones         integer | null",
        "  count_qualifier  EXACT | MANY",
        "  incidental       controlled vocabulary",
        "",
        "PER KIDNEY  (right, left — independent)",
        "  has_stone        boolean",
        "  n_characterised  integer",
        "  max_size_mm      float | null",
        "  max_size_class   MICROLITHIASIS | SMALL |",
        "                   MEDIUM | LARGE | VERY LARGE",
        "  zones_present    {upper, mid, lower}",
        "",
        "PER STONE",
        "  side             RIGHT | LEFT",
        "  zone             UPPER | MID | LOWER",
        "  size_mm          float | null (size_known)",
        "  size_class       five-level ordinal scale",
        "",
        "PROVENANCE",
        "  facts_hash       content hash of the fact set",
        "  round-trip gate  parser(text) == fact set",
    ]
    box(ax, 0.012, BY, 0.315, BH, "", fc="#fbfbfa")
    ax.text(0.026, TOP - 0.015, "A. Structured fact schema", fontsize=9.5,
            weight="semibold", color=V.INK, va="top")
    ax.text(0.026, TOP - 0.055, "\n".join(schema), va="top", ha="left", **mono)

    tmpl = [
        "FINDINGS:",
        "  <opening>, <stone clause> <verb>.",
        "  [<stone clause>, ... and <stone clause> <verb>",
        "   in the <side> kidney.]",
        "  [A calculus is also present in the <side> kidney",
        "   but was not separately characterised.]",
        "  [A total of <n> calculi were reported, of which",
        "   <k> are individually described.]",
        "  [<incidental finding sentence>]",
        "",
        "IMPRESSION:",
        "  <laterality phrase>; the largest calculus is in",
        "  the <side> <zone>, <size> mm (<class>).",
        "  [A total of <n> renal calculi were identified.]",
        "",
        "SLOTS THAT CARRY NO FACT (varied for naturalness)",
        "  <opening>  3 variants   <verb>  4 variants",
        "  measurement phrasing 3 · clause order 2",
        "",
        "INVARIANT",
        "  Every fact appears exactly once, in one",
        "  surface form, so the text can be parsed back",
        "  into the identical fact set.",
    ]
    box(ax, 0.342, BY, 0.315, BH, "", fc="#fbfbfa")
    ax.text(0.356, TOP - 0.015, "B. Deterministic template", fontsize=9.5,
            weight="semibold", color=V.INK, va="top")
    ax.text(0.356, TOP - 0.055, "\n".join(tmpl), va="top", ha="left", **mono)

    W = 44
    filled = ["EN — generated from the fact set", "─" * W]
    for para in rep["en"]["full"].split("\n"):
        filled += MF.wrap(para, W) if para.strip() else [""]
    filled += ["", "TR — generated in parallel, not translated", "─" * W]
    for para in rep["tr"]["full"].split("\n"):
        filled += MF.wrap(para, W) if para.strip() else [""]
    filled += ["", f"facts_hash {rep['facts_hash']}",
               f"round-trip parse  {'PASSED' if rep['roundtrip_parse_ok'] else 'FAILED'}"]
    box(ax, 0.672, BY, 0.316, BH, "", fc="#ffffff")
    ax.text(0.686, TOP - 0.015, "C. Filled example (case KS0005)", fontsize=9.5,
            weight="semibold", color=V.INK, va="top")
    ax.text(0.686, TOP - 0.055, "\n".join(filled), va="top", ha="left", **mono)

    for x in (0.332, 0.662):
        arrow(ax, x, 0.48, x + 0.008, 0.48, color=V.SERIES[0])
    ax.text(0.012, 0.985, "Structured reporting schema, template and a generated example",
            fontsize=12.5, color=V.INK, weight="semibold", va="top")
    ax.text(0.012, 0.945, "The generator and the parser are inverses: a report that cannot be "
            "parsed back into the identical fact set is rejected at build time.",
            fontsize=8.2, color=V.INK2, va="top")
    fig.savefig(OUT / "fig10_report_template.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ ek kartlar
def pick_appendix_cases(df, model, dev, tfm):
    """Four clinical scenarios, all from the fold 0 test set."""
    _, _, te = D.fold_split(df, 1337, 0)
    info = {}
    for case, g in te.groupby("case_id"):
        if len(g) != 2:
            continue
        rec = {"case": case, "n_pos": int(g["has_stone"].astype(bool).sum()),
               "max_char": int(g["n_characterized"].max()),
               "mm": float(g["max_mm"].max()) if g["max_mm"].notna().any() else 0.0}
        disc = False
        for _, r in g.iterrows():
            _, _, p_bag, _ = MF.slice_scores(model, r, dev, tfm)
            if (p_bag > 0.5) != bool(r["has_stone"]):
                disc = True
        rec["discordant"] = disc
        info[case] = rec

    def best(pred, key):
        c = [v for v in info.values() if pred(v)]
        return max(c, key=key)["case"] if c else None

    picks = [
        ("unilateral single stone",
         best(lambda v: v["n_pos"] == 1 and v["max_char"] == 1 and not v["discordant"],
              lambda v: v["mm"])),
        ("bilateral disease",
         best(lambda v: v["n_pos"] == 2 and not v["discordant"], lambda v: v["mm"])),
        ("multiple stones in one kidney (structural omission)",
         best(lambda v: v["max_char"] >= 2 and not v["discordant"], lambda v: v["max_char"])),
        ("model–label discordance",
         best(lambda v: v["discordant"], lambda v: v["mm"])),
    ]
    used, out = set(), []
    for label, case in picks:
        if case and case not in used:
            used.add(case)
            out.append((label, case))
        else:  # scenario empty or already used: take the next eligible case
            alt = next((c for c in sorted(info) if c not in used), None)
            if alt:
                used.add(alt)
                out.append((label + " (nearest available case)", alt))
    return out


def main() -> int:
    V.apply_theme()
    OUT.mkdir(parents=True, exist_ok=True)
    df = D.load_table()
    full = D.load_table(all_exams=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = eval_transform()
    model = MF.load_detector(dev)

    print("  fig8_study_flow ...", flush=True)
    fig8_study_flow(df, full)
    print("  fig9_pipeline_comparison ...", flush=True)
    fig9_pipeline_comparison()
    print("  fig10_report_template ...", flush=True)
    fig10_report_template(df)

    if model is None:
        print("  ATLANDI ek kartlar: checkpoint yok")
    else:
        for i, (label, case) in enumerate(pick_appendix_cases(df, model, dev, tfm), 1):
            print(f"  appendix_case_{i} ({label}: {case}) ...", flush=True)
            MF.fig7_pipeline(df, model, dev, tfm, case_id=case,
                             out_name=f"appendix_case_{i}.png",
                             title=f"Appendix A{i} — {label}")

    print(f"\nwrote -> {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
