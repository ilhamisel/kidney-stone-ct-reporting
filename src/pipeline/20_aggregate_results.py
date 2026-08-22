# -*- coding: utf-8 -*-
"""Stage 20 — aggregate run results: the mean with a 95% CI per configuration,
plus the paired comparison against the majority baseline that the protocol
requires.

Reads runs.csv only; nothing is retrained. Configurations with fewer than 25 runs
are flagged but not dropped, because preliminary five-run probes are legitimate —
they simply must not be mistaken for full-protocol results.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import pandas as pd

from common.ml import metrics as MET
from common.ml import runio as R

GROUP = ["config", "task", "model", "input", "form", "mirror", "pos", "bag_k"]


def main() -> int:
    if not R.RUNS_CSV.exists():
        sys.exit(f"ERROR: {R.RUNS_CSV} not found — run 19_train_kidney.py first")
    df = pd.read_csv(R.RUNS_CSV)
    out = []
    for key, g in df.groupby(GROUP, dropna=False):
        rec = dict(zip(GROUP, key))
        rec["n_runs"] = len(g)
        if len(g) < 25:
            print(f"WARNING: {rec['config']}/{rec['task']}: {len(g)} runs "
                  f"(< 25, not the full protocol)")
        for col in ("kidney_auc", "kidney_acc", "kidney_bal_acc", "kidney_f1",
                    "kidney_qwk", "kidney_auc_macro", "patient_acc", "patient_auc"):
            if col in g.columns and g[col].notna().any():
                m, lo, hi = MET.mean_ci(g[col].dropna())
                rec[col] = round(m, 4)
                rec[f"{col}_ci"] = f"[{lo:.4f}, {hi:.4f}]"
        if "kidney_majority" in g.columns and g["kidney_majority"].notna().all() and len(g) >= 2:
            rec["baseline_acc"] = round(g["kidney_majority"].mean(), 4)
            p = MET.paired_tests(g["kidney_acc"], g["kidney_majority"])
            rec["delta_vs_baseline"] = round(p["delta_mean"], 4)
            rec["paired_t_p"] = f"{p['t_p']:.2e}"
            rec["wilcoxon_p"] = f"{p['wilcoxon_p']:.2e}"
        rec["degenerate_runs"] = int(g.get("kidney_degenerate", pd.Series(dtype=bool)).sum())
        out.append(rec)

    summary = pd.DataFrame(out)
    summary.to_csv(R.RESULTS / "summary.csv", index=False, encoding="utf-8-sig")
    md = summary.to_markdown(index=False)
    (R.RESULTS / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwrote: {R.RESULTS / 'summary.csv'} ({len(summary)} configurations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
