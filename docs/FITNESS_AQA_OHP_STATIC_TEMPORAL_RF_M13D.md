# Milestone 13D - OHP Static vs Causal Temporal RF (train-calibrated)

This is a **source-video-group-aware, exploratory** benchmark for the two
genuinely different OHP task-specific feature schemas from PR #18.
It uses publisher OHP TRAIN and VAL only, with **no test data access**.

## Dataset successfully extracted by the owner

Original source train 1,582 / validation 339 clips all decoded, 0 failures,
2 FPS, 0.1-second annotation boundary margin, 0.50 gate on each task's
required eight landmarks:

| Model target | Train frames | Train positive frames / source videos | Validation frames | Validation positive frames / source videos | Train / val coverage |
| --- | ---: | ---: | ---: | ---: | --- |
| `error_elbows` (upper-body) | 4,253 | 101 / 56 | 886 | **25 / 14** | 31.51% / 30.11% |
| `error_knees` (lower-body) | 5,300 | 357 / 230 | 1,088 | 73 / 43 | 39.27% / 36.97% |

These are **accepted source annotation-positive frames**, not predictions or
accuracy metrics. A zero is absence of the *named* annotation, NOT a good-form
certification. The pose quality gate still abstains for 60-70% of
annotation-eligible OHP frames; metrics exclude abstentions.

The original joint 111-feature OHP coverage is only 15.49% train / 14.24%
validation. Since the task-separated gate changes the retained-frame
population, *never compare the new raw AP/F1 directly to an old model scored
on the old 111-feature population.* This PR compares two models on the
**same task-specific retained frames**, so the within-task comparison is
properly paired.

## Benchmark definitions

Each named task trains two independent classifiers with identical
source train and source validation rows:

- **Static RF:** 30 task-specific numeric coordinates and geometry features
  (`ohp-elbows-upper8-30-v1` or `ohp-knees-lower8-30-v1`).
- **Causal temporal RF:** the same 30 static features plus 12 history
  features (causal rolling mean of at most three accepted frames, and
  previous-frame delta, for the six task-specific engineered signals).
  History is computed **within source clip only**, from present/past
  frames; gaps strictly greater than 0.75 sec reset the segment.
  No interpolation over missing pose and no future frames.
  Total **42 features**, different versioned task schema.

Random Forest: `n_estimators=300`, `min_samples_leaf=2`,
`class_weight=balanced`, `random_state=42` by default.
No arbitrary threshold selection from validation.
A source-TRAIN-only `GroupShuffleSplit` partitions **whole train videos**
between temporary threshold-fitting data and calibration. The
**same** train-group partition is used for static and temporal
within a task. Both classes and sufficient annotation-positive clips
are required in each partition. Fit/calibration video groups cannot
overlap. Then two new models are refit on full source TRAIN using their
own 30/42-dimensional input schema.

Validation (untouched by fit and threshold selection) reports AP,
class prevalence, positive precision/recall, FPR, macro-F1,
source-video clip-any accepted-frame metrics at threshold .50 and the
train-selected threshold, plus a **paired** video-cluster bootstrap
of AP and macro-F1 differences (temporal minus static).
Resampling whole validation video IDs preserves correlations between
accepted frames and model pairing. Confidence intervals are
percentile intervals and include finite-sample/multi-clip uncertainty
**only**. They do **not** account for all dataset biases, repeated
validation development or lack of independent athlete IDs.

## Safe invocation (needs PR #18 or its successor merged)

PR #19 is initially a **stacked PR based on PR #18**, because
the task-feature and extractor source files do not exist on `main`
until PR #18 is merged. Checkout this branch directly; it includes
all dependencies:

```bash
git fetch origin
git switch feat/m13-ohp-static-temporal-rf-benchmark 2>/dev/null || \
  git switch --track origin/feat/m13-ohp-static-temporal-rf-benchmark
git pull --ff-only
python -m pytest -q

python -m scripts.benchmark_fitness_aqa_ohp_tasks \
  --features-dir data/features/fitness_aqa_ohp_tasks \
  --output-dir artifacts/fitness_aqa_ohp_task_rf \
  --seed 42 --trees 300 --max-gap-seconds .75 --bootstraps 500
```

**Do not re-extract OHP or download ZIP again:** this script reads the
already-generated 4 TRAIN/VAL CSVs and extraction summary only.

The default benchmark **rejects small pilot extractions**, incomplete
train/val splits, malformed CSV or nonfinite features, output-summary
mismatches, cross-split video overlap, low count of positive validation
videos, and attempts to label it as source TEST.
`--allow-pilot` is for synthetic automated fixture testing only,
not user-reported model results.

Expected Git-ignored outputs:

```text
artifacts/fitness_aqa_ohp_task_rf/
├── comparison.json
├── error_elbows_static.joblib
├── error_elbows_causal_temporal.joblib
├── error_knees_static.joblib
└── error_knees_causal_temporal.joblib
```

Models are local `joblib` files (pickle-backed); load only trusted
local artifacts, never untrusted downloaded pickles. This step **does
not integrate** models into the webcam app or suggest fitness injuries
can be prevented.

## Interpreting the first results

1. Compare **within each task** temporal versus static AP and
   train-calibrated macro-F1, then paired bootstrap
   `temporal_minus_static` AP/F1 intervals.
2. Inspect positive precision/recall and FPR; predicting all negatives
   may yield deceptively high overall accuracy due to extreme class
   imbalance (validation elbow ~2.8% positive frames, knee ~6.7%).
3. Inspect video-group calibration stability: the threshold uses
   a **single** group split. If rare elbow events make threshold
   unstable, conduct repeated train-only split/seed assessment next,
   without tuning thresholds on source validation.
4. Keep original publisher TEST data locked until model/schema/threshold
   protocol has been finalized; source splits are clip-disjoint but
   **not proven subject-disjoint**.
5. Critically: this validation split has been examined repeatedly during
   ongoing research. Bootstrap CIs cannot make it a pristine holdout.
   Refer to this as *exploratory source validation*, not an unbiased
   estimate of production performance or conference-quality independent
   generalization.
