"""Source and destination paths for the pipeline.

This repository ships code, figures and aggregate results only; the imaging data
and the label spreadsheets are not public (see `docs/data-availability.md`).
Nothing here is hard-coded to a particular machine.

Two environment variables control everything:

    KIDNEYCT_ROOT      working directory holding the stage outputs described in
                       `docs/pipeline.md`. Defaults to `./workdir` next to the
                       repository.
    KIDNEYCT_SOURCES   JSON file describing the read-only inputs (DICOM trees and
                       label spreadsheets). Defaults to `<root>/config/sources.json`.
                       See `config/sources.example.json` for the schema.

Source trees are treated as read-only; no stage writes to them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

ROOT = Path(os.environ.get("KIDNEYCT_ROOT", _REPO / "workdir")).expanduser()

PRIVATE = ROOT / "00_private"
INDEX = ROOT / "01_index"
ARCHIVE = ROOT / "02_archive"
DERIVED = ROOT / "03_derived"
LABELS = ROOT / "04_labels"
SPLITS = ROOT / "05_splits"
QA = ROOT / "06_qa"
HOLDOUT = ROOT / "07_holdout_unlabeled"
ML = ROOT / "09_ml"
LOGS = ROOT / "logs"

CONFIG = Path(os.environ.get("KIDNEYCT_SOURCES", ROOT / "config" / "sources.json")).expanduser()


def _sources() -> dict:
    if CONFIG.exists():
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    return {}


_S = _sources()
_DEFAULT_SRC = ROOT / "00_sources"


def _path(key: str, default: Path) -> Path:
    v = _S.get(key)
    return Path(v).expanduser() if v else default


# --- DICOM source roots -------------------------------------------------------
# Each entry is (root_id, path, collection label). Copies of the same study may be
# spread across several trees; the union is taken over SOPInstanceUID, so listing a
# tree twice is harmless. See `01_scan_dicom_index.py`.
DICOM_ROOTS: list[tuple[str, Path, str]] = [
    (str(r["id"]), Path(r["path"]).expanduser(), str(r["collection"]))
    for r in _S.get("dicom_roots", [])
] or [("LOCAL", _DEFAULT_SRC / "dicom", "UNSPECIFIED")]

# --- Label sources ------------------------------------------------------------
# Primary: one row per patient, per-stone zone and diameter as recorded by the
# reporting radiologist. Secondary: the free-text report used for cross-checking.
XLSX_PRIMARY = _path("xlsx_primary", _DEFAULT_SRC / "primary_labels.xlsx")
XLSX_SECONDARY = _path("xlsx_secondary", _DEFAULT_SRC / "secondary_reports.xlsx")

# Radiologist-selected slices, used to validate the index semantics (stage 07b).
SELECTED_SLICES_DIR = _path("selected_slices_dir", _DEFAULT_SRC / "selected_slices")

# Coronal PNG volumes from the earlier pipeline. Reference only; never an input to
# any modelling stage.
LEGACY_CORONAL_DIRS = {
    k: Path(v).expanduser() for k, v in _S.get("legacy_coronal_dirs", {}).items()
} or {k: _DEFAULT_SRC / "legacy_coronal" / k for k in
      ("BOBREK_TASI", "BOBREK_URETER_TASI", "URETER_TASI")}

ALL_OUTPUT_DIRS = [PRIVATE, INDEX, ARCHIVE, DERIVED, LABELS, SPLITS, QA, HOLDOUT, LOGS]


def ensure_dirs() -> None:
    for d in ALL_OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
