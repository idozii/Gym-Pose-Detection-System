# Fitness-AQA V3: Paired video-cluster uncertainty analysis

## Why

The user ran the **real-data** 103-static vs 119-causal-temporal comparison on the
publisher's Fitness-AQA validation split (984 retained frames, 206 clips).
Thresholds were calibrated on **train clips only**, keeping validation out of
threshold optimization.

| Named annotation | Metric | Static | Causal temporal |
| --- | --- | ---: | ---: |
| Knees forward | Average Precision | 0.3769 | 0.4873 |
| Knees forward | Macro-F1 | 0.6212 | 0.6997 |
| Knees forward | Error recall | 0.7850 | 0.8692 |
| Knees forward | False positive frames | 289 | 229 |
| Knees inward | Average Precision | 0.1600 | 0.2079 |
| Knees inward | Macro-F1 | 0.5558 | 0.6277 |
| Knees inward | Error recall | 0.3000 | 0.3500 |
| Knees inward | False positive frames | 86 | 42 |

These are *validation observations*, not held-out independent-subject test results.

## Run

```bash
git fetch origin
git switch --track origin/feat/fitness-aqa-paired-video-bootstrap
python -m pip install -e '.[dev]'
python -m pytest -q

# Reuse existing local RF models and validation CSV. No training or video processing.
python -m scripts.fitness_aqa_paired_bootstrap --repeats 1000 --seed 42
```

Output: `artifacts/fitness_aqa_robustness/paired_bootstrap.json`.
The report is JSON, includes static and temporal point estimates, and 2.5/97.5
percentiles of **temporal minus static** paired changes in AP, macro-F1,
error precision, error recall, and false-positive *rate*. Data and model
artifacts remain ignored by Git.

## Method

- Load models created by `scripts.compare_fitness_aqa_temporal`. Only
  load **trusted locally trained** joblib model bundles.
- Recompute static and causal temporal features on the **same 984 validation
  observations**; assert feature names, model schema, task, classes and
  previously selected train-calibrated thresholds.
- Sample **video IDs** (not frames) with replacement. For each selected
  video, include all pose-accepted frames from that video.
- Use the **same sampled video groups** for both variants (paired
  comparison); compute a temporal-minus-static difference.
- Repeat 1,000 times with fixed RNG seed, reporting percentile intervals,
  number of valid draws and number excluded because they contain only one
  target class. Hold thresholds fixed; never optimize within bootstrap.

## Interpret cautiously

The intervals estimate conditional *between-video sampling variability
within this already examined validation set*. They do **not** incorporate:
- retraining uncertainty or different train/calibration seeds;
- dependence between video IDs recorded from the same unknown participant;
- entire video intervals where MediaPipe returned **no pose** (~49.4% of
  eligible frames excluded overall);
- selection bias from comparing variants on this validation split;
- a truly untouched final test result.

A 95% paired bootstrap interval excluding zero is **not proof of**
independent-subject generalization or deployment readiness.

The recommended subsequent research work is:
1. Verify stability across several **train-only** grouped calibration seeds.
2. Audit why critical joints are lost; preserve pose rejection rates and
   compare feature extraction configurations on the same source data.
3. Freeze model, threshold and protocol before one-time publisher test
   evaluation; consider subject metadata if available.
4. For live coaching, add explicit exercise phase/rep counting and
   temporal alarm debouncing only after reliability evaluation.

`error_knees_inward=0` / `error_knees_forward=0` means the named error
was not annotated in that interval — not that the squat was perfect.
