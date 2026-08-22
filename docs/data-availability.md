# Data availability

**No patient data are included in this repository.** The imaging archive, the label
spreadsheets and every per-patient output are excluded, and `.gitignore` blocks the working
directories that hold them.

## What is here

| | |
|---|---|
| `src/` | the complete analysis code, all thirty stages |
| `figures/` | the publication figures |
| `results/` | aggregate result tables — one row per configuration, seed and fold, or per metric. No per-patient rows |
| `tests/` | unit tests, using synthetic fixtures only |

## What is not here, and why

| Excluded | Reason |
|---|---|
| DICOM archive, 16-bit HU archive, derived crops, coronal crops | patient imaging |
| Label spreadsheets, parsed labels, per-case report JSON | patient data and clinical text |
| Cross-validation splits, per-kidney prediction dumps, per-case report comparisons | one row per patient or kidney |
| Model checkpoints | trained on patient data |
| Name alias table, study-resolution table, adjudication table | keyed by patient names |

The three tables in the last row deserve a note, because the code reads them at import time.
They encode hand-verified exceptions found during dataset construction: seven spelling
differences between the label table and the folder names, five patients with two examinations
each, and three source conflicts resolved from the images or by prior radiologist adjudication.
All three are keyed by normalized patient names and are therefore identifying. They are read
from `$KIDNEYCT_ROOT/00_private/` and are absent from this repository. Each loader returns an
empty table when the file is missing, and the affected stage then reports the unmatched or
unresolved cases explicitly rather than failing silently.

## Access

The study was approved by the Institutional Ethics Committee for Non-Interventional Clinical
Research (decision 181, 10 September 2024) and conducted in accordance with the Declaration of
Helsinki; the requirement for written informed consent was waived given the retrospective
design. The ethics approval does not cover open publication of the imaging data.

An anonymized release of the dataset — de-identified image archive, structured JSON
annotations and report templates — has been prepared with an automated protected-health-
information audit (stage `15_export_public`). Requests for access should be directed to the
corresponding author and are subject to institutional approval and a data use agreement.

## Running the pipeline on your own data

The pipeline is not tied to this cohort. Point `KIDNEYCT_ROOT` at a working directory,
describe your DICOM trees and label spreadsheets in a sources file
(`config/sources.example.json` documents the schema), and run the stages in order. The label
parser expects one row per patient with per-stone zone and diameter, plus a free-text report
column used for cross-checking; the schema is described in `src/pipeline/06_parse_labels.py`.

Stages `26_evaluate_reports` and `29_error_analysis` are the exception: they read stored
prediction dumps and need no imaging at all.

## Figures

Figures that display CT data show de-identified per-kidney crops from the study cohort and
carry pseudonymous case identifiers (`KS####`) only. They are the figures submitted with the
manuscript. No identifying header, name, date or accession number appears in any of them.
