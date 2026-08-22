# Results

All model results come from patient-level stratified five-fold cross-validation repeated over
five seeds — 25 runs per configuration. Values are the mean over runs with 95% confidence
intervals from the t distribution. Comparisons against the majority baseline are paired on the
same (seed, fold) cells and assessed with both a paired t test and a Wilcoxon signed-rank test.
Seeds are fixed, `cudnn.deterministic` is enabled and `benchmark` disabled, so a repeated run
reproduces its prediction file byte for byte.

Source tables: [`results/runs.csv`](../results/runs.csv) and
[`results/summary.csv`](../results/summary.csv). Main results carry the configuration prefix
`e9_`; their counterparts on the full 394-kidney cohort carry `e3_` and `e7_`; ablations carry
`e1_`, `e4_`–`e6_` and `e8_`.

## Cohort

195 patients, 195 examinations, 390 kidneys (246 stone-positive, 144 negative). There is no
patient-level negative class — every examination in the cohort contains a stone — so detection
cannot be claimed at patient level. A genuine negative class appears only when the kidney is
the unit. 227 kidneys carry a characterized stone and form the cohort for the size and zone
tasks.

Three measurements taken while the dataset was rebuilt quantify why the earlier pipeline
failed:

| Measurement | Result |
|---|---|
| Fraction of ≥200 HU stone voxels saturated under a soft-tissue window (WL 40 / WW 400) | **83.1%** |
| The same fraction under the stone window (WL 400 / WW 1500) | **4.0%** |
| Hypothesis that the radiologist-selected index is an axial *row* index | confirmed; \|offset\| ≤ 2 in 84% against a 6.2% chance rate, binomial **p = 3.8 × 10⁻³³** |
| Isolated high-HU foci on selected rows versus random rows | 1.27×, 132/185 cases, Wilcoxon **p = 5.5 × 10⁻¹⁴** |

Kidney segmentation succeeded on all 394 kidneys and the right/left orientation self-check
passed for all of them. Median mask volume 153 cm³, median craniocaudal length 102 mm, both
within adult reference ranges; 7 kidneys are flagged for radiologist review.

## Task 1 — stone presence

390 kidneys, 78 test kidneys per fold, majority baseline 0.631.

| Model | Kidney AUC | Kidney accuracy | Balanced accuracy | Macro F1 | Patient accuracy |
|---|---|---|---|---|---|
| ResNet-18 | 0.978 (0.972–0.985) | 0.939 (0.926–0.952) | 0.932 | 0.934 | 0.975 (0.966–0.985) |
| **Hybrid** | **0.980 (0.975–0.986)** | 0.936 (0.922–0.951) | 0.928 | 0.931 | 0.975 (0.963–0.987) |

Both exceed the baseline decisively (ResNet-18 Δ = +0.308, paired t p = 1.3 × 10⁻²⁵,
Wilcoxon p = 1.2 × 10⁻⁵; hybrid Δ = +0.306, t p = 1.7 × 10⁻²⁴, Wilcoxon p = 1.2 × 10⁻⁵). None
of the 50 runs produced a degenerate prediction distribution. On the independent cohort the two
encoders are statistically indistinguishable, so architecture does not determine the outcome.

**Input representation.** The single stone window beat the three-window multichannel input
(0.978 versus 0.969 AUC), so the full protocol was run single-window. Stone presence does not
need parenchymal context; the extra channels behave as mild noise.

## Task 2 — stone size class

227 kidneys with a characterized stone. Small ≤ 5 mm (n = 46), medium 6–10 mm (n = 88),
large ≥ 11 mm (n = 93). Ordinal two-threshold formulation, majority baseline 0.419.

| Model | Accuracy | Balanced accuracy | Macro F1 | Quadratic weighted κ |
|---|---|---|---|---|
| ResNet-18 | 0.741 (0.718–0.763) | 0.698 (0.672–0.724) | 0.699 | 0.698 (0.649–0.747) |
| **Hybrid** | **0.752 (0.725–0.780)** | **0.712 (0.684–0.741)** | **0.713** | **0.726 (0.692–0.760)** |

Per-class sensitivity for the hybrid model: large 0.880, medium 0.748, small 0.500. The error
structure is favourable — 23.2% of predictions fall into an adjacent class and only 1.7% skip a
class — so the model preserves the ordering of the size axis and errs at class boundaries.

**Imaging plane (negative result).** Whether the coronal plane helps was tested on the same
folds, paired:

| Input | Accuracy | QWK | Macro F1 | Seconds per run |
|---|---|---|---|---|
| Axial (baseline) | 0.731 | 0.665 | 0.682 | 156 |
| Axial + coronal | 0.730 (Δ = −0.001, p = 0.97) | 0.665 | 0.639 | 256 |
| Coronal only | 0.640 (Δ = −0.090) | 0.569 | 0.531 | 185 |

Coronal information does not help. This is consistent with the sampling: in-plane spacing is
0.94 mm on axial, whereas one axis of a coronal reformat is resampled from the slice interval.
The two-view arm was removed from the codebase; the advantage of the coronal plane is position,
not size.

## Task 3 — intrarenal zone

227 kidneys, multi-label (upper n = 35, mid n = 136, lower n = 127).

End-to-end formulations do not learn the task. These four rows are ablations and were run on
the full 394-kidney cohort:

| Formulation | Macro AUC | Runs |
|---|---|---|
| End-to-end, axial (ResNet-18) | 0.553 (0.528–0.579) | 25 |
| End-to-end + slice z-position (ResNet-18) | 0.575 (0.547–0.604) | 25 |
| End-to-end, coronal (ResNet-18) | 0.601 (0.530–0.671) | 5 |
| End-to-end, coronal (hybrid) | 0.584 (0.479–0.688) | 5 |

The cause is architectural, not an absence of signal. Zone is a position on the z axis; the
in-plane appearance of a stone does not differ between zones, and log-sum-exp pooling is
permutation-invariant, so the encoder never sees where in the stack a slice came from.

**Two-stage formulation.** Train the stone-presence detector (validation AUC 0.989), take the
normalized within-kidney position of its highest-scoring slice, and map that position to a zone
with two thresholds searched on the training fold. There is no gradient training for zone.

| Model | Macro AUC | Three-class accuracy | Baseline | Δ | Paired t p |
|---|---|---|---|---|---|
| ResNet-18 | 0.776 (0.757–0.796) | 0.694 (0.675–0.713) | 0.517 | +0.177 | 4.7 × 10⁻¹¹ |
| **Hybrid** | **0.777 (0.751–0.803)** | 0.694 (0.670–0.718) | 0.517 | +0.177 | 3.3 × 10⁻⁹ |

Per-zone AUC for the hybrid model: upper 0.821, mid 0.685, lower 0.816. The learned thresholds
converge on 0.353 and 0.689 (hybrid) and 0.340 and 0.675 (ResNet-18), against geometric thirds
at 0.333 and 0.667 — an independent confirmation of the kidney-relative zone definition, with
no systematic frame offset.

**Attainable ceiling.** Applying the same threshold rule to stone positions measured directly
from the images, on the 103 kidneys with a single stone and a single reported zone:

| Method | Three-class accuracy |
|---|---|
| Measured position, geometric 1/3–2/3 thresholds | 0.767 |
| Measured position, thresholds fitted to the data | **0.777 (upper bound)** |
| Two-stage, hybrid (learned detector) | **0.736 ± 0.023** |
| Two-stage, ResNet-18 | 0.728 ± 0.020 |

The learned detector reaches 94.8% of what perfect localization attains. That the ceiling
itself sits at 0.78 means the residual error is disagreement between radiological zone naming
and geometric thirds, which no image model can close.

## Report-level performance

Turkish and English reports are generated in parallel from one canonical fact set; neither is a
translation of the other. A parser that is the exact inverse of the generator is applied to
every report at build time. All 197 Turkish and 197 English reference reports reproduced their
fact set exactly, and so did all 195 + 195 reports generated from the model's own predictions
(390 of 390). Report-level errors therefore originate entirely in the vision models.

| Measure | Hybrid | ResNet-18 |
|---|---|---|
| Per-kidney stone presence, P / R / F1 | 0.939 / 0.962 / **0.950** | 0.946 / 0.958 / 0.952 |
| Laterality, four-class accuracy | 0.876 (0.857–0.895) | 0.881 (0.848–0.914) |
| Maximum size class, accuracy | 0.737 (0.725–0.748) | 0.723 (0.690–0.755) |
| Zone, accuracy | 0.728 (0.694–0.762) | 0.721 (0.684–0.757) |
| Triplet micro F1, reduced reference | **0.533** | 0.507 |
| Hallucination rate, reduced reference | 0.458 | 0.484 |
| Omission rate, reduced reference | 0.476 | 0.502 |
| Triplet micro F1, complete reference | 0.445 | 0.434 |
| Omission rate, complete reference | 0.637 | 0.647 |

Two observations follow. Field accuracies compound multiplicatively — a triplet is correct only
if side, zone and size class are simultaneously correct, and 0.96 × 0.73 × 0.74 ≈ 0.52 predicts
the measured 0.533 almost exactly. And the rise in omission from 0.476 to 0.637 under the
complete reference is structural: the current output expresses at most one stone per kidney, so
additional stones in the same kidney cannot be represented at all.

Where the composite error comes from is decomposed in [`error-analysis.md`](error-analysis.md).

## Effect of the cohort definition

The independence criterion — one examination per patient — was tested by running the same
configurations on both cohorts. No main metric moved by more than 0.027 and no result changed
direction or significance:

| Task / model | Metric | 390 kidneys | 394 kidneys | Δ |
|---|---|---|---|---|
| Presence / hybrid | kidney AUC | 0.980 | 0.983 | −0.003 |
| Presence / ResNet-18 | kidney AUC | 0.978 | 0.972 | +0.006 |
| Size / hybrid | QWK | 0.726 | 0.729 | −0.003 |
| Size / ResNet-18 | accuracy | 0.741 | 0.736 | +0.005 |
| Zone / hybrid | macro AUC | 0.777 | 0.782 | −0.005 |
| Zone / ResNet-18 | macro AUC | 0.776 | 0.767 | +0.009 |

One qualitative change is worth recording: on the full cohort the hybrid encoder led on all
three tasks, whereas on the independent cohort the two are indistinguishable. Part of the
earlier gap came from confidence intervals narrowed by pseudo-replication, which supports the
cohort correction.

## Auxiliary component — stone candidate generator

The mask-constrained heuristic generator produces a median of two candidates per case and
matches 67.8% (242/352) of reported stones, with a clean gradient by size class: 0.18 at ≤3 mm,
0.37 at 4–5 mm, 0.78 at 6–10 mm, 0.88 at 11–19 mm and 0.97 at ≥20 mm. It is a candidate
generator, not a detector, and is not used as model input.

## Limitations

Composite report agreement (triplet F1 0.533) is well below the individual field accuracies;
the system is a draft generator requiring radiologist verification, not a deployable one. The
output expresses at most one stone per kidney. There is no patient-level negative class, and
the kidney-level negative class rests on "not reported", which is strong but not definitive
evidence of absence. Size classes derive from reported millimetres rather than from
measurements on the images. The attainable ceiling of the zone task is bounded near 0.78 by
label noise. Ablations on input window, imaging plane and end-to-end zone were run on the full
394-kidney cohort rather than the modelling cohort; their qualitative conclusions are unaffected
by the 1% difference. The coronal arm was run with a single seed and does not carry the same
statistical power. No external validation, reader study or per-stone measurement was performed.
