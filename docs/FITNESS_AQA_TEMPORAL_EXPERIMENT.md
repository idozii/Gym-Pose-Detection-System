# Fitness-AQA V2: train-only threshold calibration and causal temporal-feature ablation

## Why this experiment?

The owner's real source validation (984 pose-accepted frames) gave:

| Task | RF threshold 0.5 macro-F1 | Recall | AP (ranking quality) | Prevalence |
| --- | ---: | ---: | ---: | ---: |
| Knee forward annotation | 0.5841 | 0.2897 | 0.3769 | 0.2175 |
| Knee inward annotation | 0.4891 | 0.0000 | 0.1600 | 0.0407 |

Choosing thresholds 0.35 and 0.20 on the **same validation data**
improved apparent macro-F1, but makes evaluation optimistic for threshold selection.

## New controlled comparison

1. Use only existing **train** CSV (4,309 accepted observations) to split
   original source train videos into disjoint *fit* and *calibration* clips
   with `GroupShuffleSplit` (seed 42, 25% video groups for calibration).
2. Fit a temporary RF on fit clips, predict calibration clips and select
   a threshold using **train-only** calibration macro-F1 (predeclared
   threshold grid 0.05–0.95 in increments of 0.05).
3. Fit final RF on the entire source train data, using that threshold.
4. Report final performance on the **untouched publisher validation clips**.
   The source test clips are never read.
5. Perform above steps for two features variants and both target annotations:
   - `static`: original 103 feature columns.
   - `causal_temporal`: 103 original plus 16 derived columns
     (rolling mean over last 3 accepted frames and previous-frame delta for
     eight geometry/orientation signals). Rolling windows reset on gaps >0.75 s
     and **never** look into future frames or mix videos.

## Run locally using the CSVs and models you already produced

```bash
git fetch origin
git switch --track origin/feat/fitness-aqa-grouped-calibration-temporal-ablation
python -m pip install -e '.[dev]'
python -m pytest -q
python -m scripts.compare_fitness_aqa_temporal
```

No video ZIP access or extraction needed. Four models and comparison metrics
will be written under `artifacts/fitness_aqa_temporal` (Git-ignored).
The CLI prints AP, macro-F1, positive-class precision and recall, false positives,
and group-trained thresholds on publisher validation. Full JSON includes
fixed-0.5 and train-calibrated results, per-video *any accepted frame positive*
aggregation, fit/cal video counts, and clear limitations.

## Interpretation and next work

Treat validation differences as comparative research, **not** unbiased generalization
if variant choice is based on validation. Clip-ID disjoint does not establish
subject-ID disjoint; frame samples within one clip correlate. Temporal signals
do not solve low MediaPipe pose coverage (~50.6% on eligible train+val frames).
A zero label means only that this specific error was not annotated.

If train-only calibration improves error recall without excessive false alerts,
keep that strategy as an ablation. If temporal features do not improve AP over
static features, prioritize pose-quality failure analysis and more informative
joint geometry (including camera/view effects), not a deeper network by default.
Choose an operating point based on project requirements before the final locked
test evaluation. Model bundles from this offline comparison are **not**
drop-in compatible with the old binary-correctness webcam CLI.
