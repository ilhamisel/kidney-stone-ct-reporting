# Pipeline

Thirty numbered stages, each idempotent and each writing an auditable log. Stages 01–18 build
the dataset from DICOM, 19–23 do the modelling, 24–29 produce the figures, the report-level
evaluation and the error analysis.

Run any stage from the repository root with `src` on the path:

```bash
PYTHONPATH=src python src/pipeline/<stage>.py [--flags]
```

## Stages

| Stage | Purpose | Environment |
|---|---|---|
| `00_env_check` | verify interpreter, packages and source reachability | data |
| `01_scan_dicom_index` | index every DICOM instance; union over SOPInstanceUID across duplicate trees | data |
| `02_match_patients` | match folder names to label rows through the three-layer name normalizer | data |
| `03_select_series` | choose the axial series per study from `ImageOrientationPatient`, thickness and extent; log the reason | data |
| `04_build_archive_png` | lossless 16-bit HU archive (`png_uint16 = HU + 4096`) plus a geometry-carrying NIfTI mirror | data |
| `05_build_derivatives` | 8-bit model-ready derivatives under the stone window (WL 400 / WW 1500) | data |
| `06_parse_labels` | parse per-stone zone and diameter; adjudicate source conflicts | data |
| `07_map_selected_slices` | map radiologist-selected slice indices onto the archive | data |
| `07b_verify_index_semantics` | test three hypotheses for what the selected index means | data |
| `07c_validate_selected_rows` | test whether selected rows carry more high-attenuation foci than random rows | data |
| `08a_fix_nifti_affine` | correct the NIfTI affine — **mandatory before 08** | data |
| `08_segment_kidneys` | TotalSegmentator kidney masks, per-kidney crops, right/left self-check | segmentation |
| `08b_validate_zone_frame` | measure stone position from the images and test it against the reported zone | data |
| `09_detect_stone_candidates` | heuristic stone candidate generator, constrained by the kidney mask | data |
| `10_generate_reports` | deterministic Turkish and English reports from one canonical fact set | data |
| `11_assemble_report_json` | one JSON per case: labels, targets, imaging, reports, splits, provenance | data |
| `12_make_splits` | patient-level stratified folds, written to disk | data |
| `13_qa_checks` | quality gates over the whole dataset | data |
| `14_contact_sheets` | per-case contact sheets for visual review | data |
| `15_export_public` | audit and export the publishable subtree, with a PHI scan | data |
| `16_build_holdout` | assemble the unlabeled hold-out cohort | data |
| `17_verify_cohort_vs_previous` | cross-check the cohort against an earlier pipeline | data |
| `18_build_kidney_dataset` | the single kidney-level table used by every modelling stage | data |
| `19_train_kidney` | training: `--task {has_stone,size3,zone} --input {stone,mw3,cor}` | training |
| `20_aggregate_results` | aggregate runs into summary tables | training |
| `21_repair_crop_meta` | rebuild crop metadata from the masks | data |
| `22_build_coronal_crops` | coronal crop stacks, isotropic in z | data |
| `23_zone_two_stage` | two-stage zone: detector position plus thresholds fitted on the training fold | training |
| `24_make_figures` | result and data figures | training |
| `25_make_report_samples` | worked example reports | training |
| `26_evaluate_reports` | **report-level evaluation — the primary metric.** No training required | training |
| `27_build_manuscript` | Markdown to Word, with revision text marked | training |
| `28_make_paper_figures` | flow diagram, pipeline comparison, template figure, appendix cards | training |
| `29_error_analysis` | error decomposition. No training required | training |

## Data contract

The kidney-level table is the single source of truth for modelling. The `primary_exam` column
selects one examination per patient — the earliest by date, a criterion independent of the
outcome, so the cohort is not selected on the result. `load_table()` applies that filter by
default; `load_table(all_exams=True)` returns the full cohort for descriptive statistics.

| | Modelling cohort | Full cohort |
|---|---|---|
| Patients | 195 | 195 |
| Examinations | 195 | 197 |
| Kidneys | 390 | 394 |
| Stone-positive / negative kidneys | 246 / 144 | 248 / 146 |
| Kidneys with a characterized stone | 227 | 229 |

Primary model input is the per-kidney axial crop under the stone window. Coronal crops exist
and were tested; they help neither the size task nor the zone task enough to justify their
cost, and are retained only for the ablations.

## Ordering constraints

- `08a` must run before `08`. The initial NIfTI affine mapped the array axes to the wrong
  direction cosines and assumed RAS rather than LPS, which swapped right and left. After the
  fix, the segmentation right/left self-check passes for all 394 kidneys.
- `21` must run after `09 --write`, which used to overwrite the crop blocks written by `08`.
- Any stage producing model scores must feed the input with mirroring enabled. Left kidneys
  fed without mirroring produce meaningless scores, and the failure is silent.

## Conventions

- Axis convention: for the corrected affine, axis 0 runs anterior to posterior and axis 1 runs
  toward the patient's left. Every component that consumes this convention is verified
  separately — verifying it once at the source is not sufficient.
- Class boundaries are half-open. Size classes are `mm < 6` (small), `6 ≤ mm < 11` (medium)
  and `mm ≥ 11` (large). All recorded diameters are whole millimetres.
- Zone boundaries are equal thirds of the craniocaudal extent of the kidney itself, not of the
  abdomen, so the definition is independent of patient positioning and body habitus.
