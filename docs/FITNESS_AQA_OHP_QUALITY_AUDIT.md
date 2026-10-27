# Milestone 13B prerequisite - OHP pose gate failure audit

## Owner's **complete** source TRAIN+VALIDATION extraction

On the publisher's Fitness-AQA overhead press clips, `ohp-pose-111-v1`,
2 frames/sec, boundary margin 0.1s and 12 critical joints visibility >=0.50:

| Metric | Train | Validation |
| --- | ---: | ---: |
| Source videos decoded | 1,582 | 339 |
| Decode failures | 0 | 0 |
| Eligible sampled frames (after boundary filtering) | 13,496 | 2,943 |
| Accepted poses | 2,090 | 419 |
| Accepted fraction | **15.49%** | **14.24%** |
| Accepted `error_elbows` positives | 60 | **17** |
| Accepted `error_knees` positives | 117 | **26** |

Original test split remains LOCKED (339 clips). The production gate requires
shoulders, elbows, wrists, hips, knees, and ankles simultaneously reliable.
An OHP form classifier with only **17 accepted positive elbow frames**
in validation risks severe sample selection and statistical uncertainty.
**Do not claim a reliable OHP model benchmark just yet.**

## Why a joint-specific audit first?

The production OHP feature extractor accepts only if all 12 MediaPipe
upper and lower body critical joints exceed visibility 0.50. Losing legs or
wrists can disproportionately reject frames even though other parts of the
person are detectable. To identify the bottleneck without casually relaxing
production safeguards, compare visibility acceptance on the **same** sampled
frames and count retained *named-error* positives.

The audit is exploratory; these choices only test **candidate admission**,
not whether the omitted/low-visibility joints yield accurate pose geometry.

## Run on your already-downloaded 907-MB OHP ZIP

```bash
git fetch origin
git switch --track origin/feat/m13-ohp-coverage-failure-audit
python -m pytest -q

python -m scripts.audit_fitness_aqa_ohp_coverage \
  --clips-per-split 48 \
  --sample-fps 2 \
  --seed 42 \
  --boundary-margin 0.1
```

This samples 48 training videos and 48 validation videos in four
source-annotation strata: both named errors, elbows-only, knees-only and
neither. It opens **one** source train/val clip at a time and deletes
the temporary clip immediately. Original TEST video is not opened.

The resulting local-only JSON appears at:

```text
artifacts/fitness_aqa_ohp/pose_quality_audit.json
```

Outputs are Git-ignored. It logs: eligible frames, no-pose rejection,
invalid geometry, critical-joint visibility rejection, per-joint low-
visibility occurrences, breakdown by annotated source-clip strata,
and four candidate admissions alongside production:

| Candidate | Joint visibility rules | Interpretation |
| --- | --- | --- |
| `all12_050_production` | All 12 critical joints >=0.50 | Currently trained/extracted features |
| `all12_035` | All 12 >=0.35 | Relax all, unknown feature quality |
| `all12_025` | All 12 >=0.25 | Even more relaxed, high reliability risk |
| `upper8_050_elbow_candidate` | Shoulders, elbows, wrists, hips >=0.50; no threshold on knees/ankles | Diagnostic elbow-task candidate |
| `lower8_050_knee_candidate` | Shoulders, hips, knees, ankles >=0.50; no threshold on elbows/wrists | Diagnostic knee-task candidate |

*Missing pose* and *invalid geometry* cannot be fixed by changing visibility
cutoffs. Candidate policies may admit frames for which features are
untrustworthy. Each policy reports additional elbow-positive/knee-positive
and negative frames and per-task acceptance fractions.

We deliberately stratify clips to include rare error-positive videos; the
audit sample is **NOT a prevalence-representative sample** of all publisher
videos. A positive annotation describes that specific error interval only.

## Decision rule

1. Diagnose which body joints cause the largest number of rejections.
2. If most OHP elbow-positive frames are rejected for knees/ankles, test
   elbow-task-specific gating while retaining trustworthy elbow and wrist
   joints. For knee error, preserve knee and ankle reliability.
3. Any new gate requires **re-extraction of train/validation** and an
   independently calibrated static/causal-temporal RF comparison on matched
   retained frames. Admission counts alone are **not** accuracy.
4. Keep source TEST untouched until a single protocol is frozen.

If no candidate recovers usable error-positive frames, consider new
camera viewpoints, a different pose detector, or study the joint
visibility distribution before adding model complexity. Model architecture
cannot recover labels from frames that were never accepted.

All views, joint geometry and named labels here are not clinically
validated or a basis for injury-prevention claims.
