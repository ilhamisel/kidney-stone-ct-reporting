# -*- coding: utf-8 -*-
"""Stage 25 — worked example cases.

Selects three cases and writes a one-page Markdown document for each:
  - the structured fact set, which is what the model is asked to predict
  - the generated Turkish and English report text, produced in PARALLEL from the
    same facts rather than by translating one into the other
  - the round-trip gate: whether parsing the text back yields the identical facts
  - the model's own prediction for those kidneys, read from the held-out dumps
  - any figure belonging to the case

The cases are not chosen at random. One carries bilateral disease together with a
count ambiguity, one an adjudicated conflict between the two label sources, and
one a zone label that disagrees with the image. They are the difficult edges of
the dataset, and a showcase that hid them would be misleading.
"""
from __future__ import annotations

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

import numpy as np
import pandas as pd

from common.ml.runio import PREDS
from common.paths import LABELS, ROOT

OUT = ROOT / "10_examples" / "reports"
ZONE_EN = {"LOWER": "lower", "MID": "mid", "UPPER": "upper", "UNKNOWN": "—"}
SIDE_EN = {"LEFT": "left", "RIGHT": "right"}


def model_preds(kidney_ids: list[str]) -> dict[str, dict]:
    """Collect the stored has_stone and two-stage zone predictions.

    Every kidney is in a test fold exactly once per seed, so all of these are
    held out; the values are averaged over seeds.
    """
    out: dict[str, dict] = {}
    for f in PREDS.glob("e9_hs_sst_has_stone_s*_f*.csv"):
        d = pd.read_csv(f)
        for _, r in d[d["kidney_id"].isin(kidney_ids)].iterrows():
            # for the binary task prob is a scalar; pandas reads it as a float
            v = r["prob"]
            out.setdefault(r["kidney_id"], {}).setdefault("p_stone", []).append(
                float(json.loads(v) if isinstance(v, str) else v))
    for f in PREDS.glob("e9_2stage_sst_zone_s*_f*.csv"):
        d = pd.read_csv(f)
        for _, r in d[d["kidney_id"].isin(kidney_ids)].iterrows():
            out.setdefault(r["kidney_id"], {}).setdefault("z_hat", []).append(float(r["z_hat"]))
    return out


def case_md(case_id: str, note: str, preds: dict) -> str:
    doc = json.loads((LABELS / "reports" / f"{case_id}.json").read_text(encoding="utf-8"))
    dem, lab, rep = doc["demographics"], doc["labels"], doc["reports"]
    coh = doc["cohort"]
    d8 = str(coh.get("study_date") or "")
    coh_s = (f"{coh.get('collection', '—')} · examination {d8[:4]}-{d8[4:6]}-{d8[6:]}"
             f"{' (assignment unverified)' if not coh.get('study_assignment_verified') else ''}")
    L = [f"# {case_id} — worked example", "", f"> **Why this case:** {note}", ""]
    L += [f"- **Patient:** {dem['age_years']} years, {dem['sex']}  ·  **cohort:** {coh_s}",
          f"- **Declared stone count:** {lab['stone_count_declared']} · "
          f"**individually characterized:** {lab['stone_count_listed']} · "
          f"**qualifier:** `{lab['count_qualifier']}`",
          f"- **facts_hash:** `{rep['facts_hash']}` · "
          f"**round-trip gate:** {'PASSED' if rep['roundtrip_parse_ok'] else 'FAILED'}", ""]

    L += ["## Structured facts (the model's target)", "",
          "| Kidney | stone | characterized | largest | class | zones | model P(stone) "
          "| model ẑ → zone |",
          "|---|---|---|---|---|---|---|---|"]
    for side in ("right", "left"):
        k = doc["targets"]["kidneys"][side]
        kid = f"{case_id}_{side[0].upper()}"
        zs = [ZONE_EN[z.upper()] for z, v in k["zones_present"].items() if v] or ["—"]
        pm = preds.get(kid, {})
        p = f"{np.mean(pm['p_stone']):.3f}" if pm.get("p_stone") else "—"
        z = f"{np.mean(pm['z_hat']):.2f}" if pm.get("z_hat") else "—"
        L.append(f"| {SIDE_EN[side.upper()]} | {'yes' if k['has_stone'] else 'no'} | "
                 f"{k['n_characterized']} | "
                 f"{('%.0f mm' % k['max_size_mm']) if k['max_size_mm'] else '—'} | "
                 f"{k['max_size_class'] or '—'} | {', '.join(zs)} | {p} | {z} |")
    L += ["", "*P(stone): the has_stone model, averaged over 5 seeds; this kidney was in "
          "the test fold under every seed. ẑ: the within-kidney position from the two-stage "
          "zone estimate (0 = caudal, 1 = cranial).*", ""]

    # Model-label disagreement, surfaced rather than hidden: it marks the
    # ambiguous edges of the dataset.
    for side in ("right", "left"):
        k = doc["targets"]["kidneys"][side]
        pm = preds.get(f"{case_id}_{side[0].upper()}", {})
        if not pm.get("p_stone"):
            continue
        p = float(np.mean(pm["p_stone"]))
        if (p > 0.5) != bool(k["has_stone"]):
            L += [f"> **Model-label disagreement ({SIDE_EN[side.upper()]}):** label "
                  f"`has_stone={k['has_stone']}`, model P = {p:.3f}. "
                  + ("The label comes from free text or adjudication rather than from an "
                     "individually characterized stone (n_characterized = 0). This is a "
                     "concrete instance of the 'not reported does not mean absent' "
                     "limitation."
                     if k["has_stone"] and not k["n_characterized"] else
                     "The disagreement stands despite a characterized stone, and is worth "
                     "review."), ""]

    if lab["stones"]:
        L += ["## Stones characterized individually in the report", "",
              "| # | side | zone | size | class | zone inferred |",
              "|---|---|---|---|---|---|"]
        for s in lab["stones"]:
            L.append(f"| {s['stone_index']} | {SIDE_EN.get(s['side'], s['side'])} | "
                     f"{ZONE_EN.get(s['zone'], s['zone'] or '—')} | "
                     f"{('%.0f mm' % s['size_mm']) if s['size_known'] else 'unknown'} | "
                     f"{s['size_class'] or '—'} | {'yes' if s['zone_inferred'] else 'no'} |")
        L.append("")

    if lab.get("conflicts") or lab.get("adjudication"):
        L += ["## Source conflict and adjudication", ""]
        for c in lab.get("conflicts") or []:
            L.append(f"- **conflict:** {json.dumps(c, ensure_ascii=False)}")
        adj = lab.get("adjudication")
        if adj:
            L.append(f"- **resolution:** {json.dumps(adj, ensure_ascii=False)}")
        L.append("")

    L += ["## Generated report — Turkish", "", "```", rep["tr"]["full"], "```", "",
          "## Generated report — English", "", "```", rep["en"]["full"], "```", ""]
    orig = rep.get("original_report_tr")
    if orig:
        if isinstance(orig, dict):
            names = {"rapor_bulgular": "Findings", "rapor_sonuc": "Impression",
                     "rapor_tam": "Primary source record"}
            txt = "\n\n".join(f"{names.get(k, k)}:\n{str(v).strip()}"
                              for k, v in orig.items() if str(v).strip())
        else:
            txt = str(orig).strip()
        L += ["## Source report text (radiologist)", "", "```", txt, "```", "",
              "*The generated text is not a copy of the source: it is rendered from the "
              "canonical fact set by the template, and the source is used only to verify "
              "the facts.*", ""]

    imgs = sorted((ROOT / "10_examples" / "figures").glob(f"case_{case_id}*.png"))
    if imgs:
        L += ["## Image", ""] + [f"![{p.stem}](../figures/{p.name})" for p in imgs] + [""]
    L += ["---", f"*Source: `04_labels/reports/{case_id}.json` · template "
          f"`{rep['template_version']}` · generator seed {rep['generator_seed']}*"]
    return "\n".join(L)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [
        ("KS0005", "Bilateral disease with a count ambiguity (5 stones declared, 1 "
                   "characterized) and an ambiguous free-text laterality, resolved to "
                   "bilateral by adjudication."),
        ("KS0029", "A source conflict: the primary table says right, the report text says "
                   "left. Resolved from the images, which show no focus at or above 300 HU "
                   "on the right. The case also carries a 35 mm staghorn calculus."),
        ("KS0161", "An instance of zone label noise: the report states upper, while the "
                   "stone lies in the lower band of the image. This is the kind of case "
                   "that bounds the attainable ceiling of the zone task."),
    ]
    kids = [f"{c}_{s}" for c, _ in cases for s in "RL"]
    preds = model_preds(kids)
    for case_id, note in cases:
        p = OUT / f"{case_id}.md"
        p.write_text(case_md(case_id, note, preds), encoding="utf-8")
        print(f"  wrote: {p.name}")
    idx = ["# Worked example cases", "",
           "Three cases at the difficult edges of the dataset. Each page shows the "
           "structured facts, the generated Turkish and English report text, and the "
           "model's own prediction side by side.", ""]
    idx += [f"- [{c}]({c}.md) — {n}" for c, n in cases]
    (OUT / "README.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    print(f"\nwrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
