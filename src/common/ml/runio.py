"""Run I/O: an interruption-tolerant result table plus per-kidney prediction dumps.

A run is keyed by (config, task, model, input, form, mirror, seed, fold). A completed
run appears in runs.csv and is skipped on restart, so a 25-run configuration can be
interrupted and resumed without repeating work. Predictions are written per kidney so
that aggregation and report-level evaluation run without retraining anything.
"""
from __future__ import annotations

import csv
from pathlib import Path

from common.paths import ML

RESULTS = ML / "results"
PREDS = RESULTS / "preds"
CKPT = ML / "checkpoints"
MLLOGS = ML / "logs"
RUNS_CSV = RESULTS / "runs.csv"
KEY_FIELDS = ("config", "task", "model", "input", "form", "mirror", "seed", "fold")


def ensure_ml_dirs() -> None:
    for d in (RESULTS, PREDS, CKPT, MLLOGS):
        d.mkdir(parents=True, exist_ok=True)


def load_done(csv_path=RUNS_CSV, key_fields=KEY_FIELDS):
    """Return the set of completed run keys and the existing rows."""
    p = Path(csv_path)
    if not p.exists():
        return set(), []
    with open(p, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    return {tuple(r[k] for k in key_fields) for r in rows}, rows


def save_rows(rows, csv_path=RUNS_CSV) -> None:
    """Rewrite the whole table (idempotent). Called after every fold.

    The column set is the union over all rows: tasks report different metrics
    (size3 has a quadratic weighted kappa but no AUC, for instance) and missing
    cells are simply left empty.
    """
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def preds_path(config: str, task: str, seed: int, fold: int) -> Path:
    return PREDS / f"{config}_{task}_s{seed}_f{fold}.csv"
