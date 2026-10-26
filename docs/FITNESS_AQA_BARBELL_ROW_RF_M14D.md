# Milestone 14D — Barbell Row Static RF Baselines (publisher train+val only)

The owner completed M14C's **full, real-data** Barbell Row torso4 extraction:
132 tests passed; 12,729 distinct original TRAIN and 2,736 distinct
VALIDATION image IDs decoded, with 0 decoder failures, 0 source TEST
images decoded and 0 exact-image cross-task split conflicts.

| Publisher task | Train accepted/source | Train positive accepted | VAL accepted/source | VAL positive accepted |
| --- | --- | ---: | --- | ---: |
| Torso angle error | 8,333/11,881 = **70.14%** | **758** | 1,789/2,564 = **69.77%** | **200** |
| Lumbar error | 7,273/10,360 = **70.20%** | **1,023** | 1,578/2,227 = **70.86%** | **251** |

M14A's old eight-joint pose extractor achieved 4–5% coverage, and
only 8 positive lumbar validation samples. The new shoulder/hip
**four-joint** extractor genuinely uses ONLY those four reliable
landmarks (20-feature schema) and keeps far more original-label
positives. This is evidence of better POSE ADMISSION, NOT better
classification or direct measurement of lumbar spine curvature.

## Training and evaluation protocol

This milestone adds **two separate static Random Forest classifiers**,
one for publisher `torso_angle` binary labels and the other for
`lumbar`. All code operates on **existing M14C CSVs and extraction
summary**; no images are opened, no MediaPipe or ZIP downloads required.

For each target independently:
1. Validate that the source extraction is **complete** (not pilot),
   exactly the original source train/val split image counts,
   correct versioned 20 columns/labels, count consistency and zero
   image-decoding failures.
2. Reject train/val IMAGE-ID overlap and overlap in source
   `image_id.rsplit('_',1)[0]` **candidate prefixes**. Important:
   filename prefixes are NOT verified source video/athlete identifiers.
   This reduces one possible leakage mode, but does not establish
   subject independence.
3. Hold out approximately 25% of source TRAIN candidate prefix groups,
   selecting a split with multiple positive-label groups in both
   train-fitting and calibration sets. Fit a temporary RF, select
   the macro-F1 classification threshold solely from **TRAIN
   calibration**, then refit final RF on all original TRAIN accepted
   images. Neither fit nor threshold choice sees original VAL labels.
4. Evaluate original VAL at both the default **0.50** threshold and
   the train-calibrated threshold. Report AP, source-label positive
   prevalence as the random-ranking AP reference, macro-F1, precision,
   recall, FPR and counts of true/false positives/negatives.
5. Bootstrap VAL candidate-prefix blocks to quantify sensitivity
   to repeated related images, while **explicitly not interpreting
   this as a verified subject-level or video-level confidence
   interval**. The original val set has been looked at across
   several milestones, so this remains exploratory analysis.

No video temporal model, filename-derived sequence, webcam inference
integration, external dataset or publisher TEST is involved.
MediaPipe shoulder+hip-only landmarks do **not** directly observe
lumbar spinal curvature and cannot establish physical spinal safety.

## Run on the already-extracted full M14C CSVs

```bash
git fetch origin
git switch feat/m14-barbell-row-torso4-static-rf 2>/dev/null || \
  git switch --track origin/feat/m14-barbell-row-torso4-static-rf
git pull --ff-only
python -m pytest -q

python -m scripts.benchmark_fitness_aqa_barbell_row_torso4 \
  --features-dir data/features/fitness_aqa_barbell_row_torso4 \
  --output-dir artifacts/fitness_aqa_barbell_row_torso4_rf \
  --seed 42 --trees 300 --bootstraps 500
```

Output is Git-ignored:

```text
artifacts/fitness_aqa_barbell_row_torso4_rf/
├── comparison.json
├── barbell_row_torso_angle_torso4_static_rf.joblib
└── barbell_row_lumbar_torso4_static_rf.joblib
```

Only load `.joblib` files from your own trusted runs: joblib uses pickle
and is unsafe with untrusted model artifacts.

Do NOT optimize feature engineering, tune seed/threshold or select final
model variants using VAL results, then report the same VAL as a pristine
unbiased holdout. Keep source publisher TEST locked until a final protocol
is frozen. A zero source error label means only the *named* error is
not annotated, not globally sound Barbell Row form.

## Next decision

Once the owner's real-data `comparison.json` is available, inspect
source-label AP above/below accepted-source prevalence, macro-F1, recall,
precision, FPR, validation bootstrap interval and acceptance rate.
If prediction quality is weak, stop short of deploying real-time Row
warnings; shift to model reliability/data provenance review.

The overall roadmap continues to M15 model registry/inference,
M16 real-time application and M17 final evaluation/demo, **subject
to not advertising experimental weak detectors as dependable coaching**.
