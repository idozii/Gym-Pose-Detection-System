# Fitness-AQA: Train-only calibration seed stability

## Existing evidence

The user's real-data paired video bootstrap (1,000 resamples, 206 validation videos)
of **119-feature causal temporal** versus **103-feature static** Random Forest:

| Metric, temporal minus static | Knees forward: 95% video-bootstrap interval | Knees inward: interval |
| --- | --- | --- |
| Average precision | +0.1104 [0.0649, 0.1686] | +0.0478 [0.0009, 0.1015] |
| Macro-F1 | +0.0785 [0.0580, 0.1002] | +0.0719 [0.0176, 0.1253] |
| Recall | +0.0841 [0.0290, 0.1455] | +0.0500 [-0.0667, 0.1634] |
| False-positive rate | -0.0779 [-0.1024, -0.0516] | -0.0466 [-0.0669, -0.0303] |

These are **conditional** uncertainty intervals on an already examined validation set.
Knees-inward recall interval includes zero, so no reliable recall improvement is established.

## Next: vary the train-group calibration seed (without re-extracting videos)

```bash
git fetch origin
git switch --track origin/feat/fitness-aqa-multiseed-calibration
python -m scripts.fitness_aqa_multiseed_calibration \
  --calibration-seeds 42 52 62 72 82 --forest-seed 42
```

Only uses the 4,309 train frames and 984 validation frames from your existing
CSV files. Reads **no test files**. Output is Git-ignored:
`artifacts/fitness_aqa_robustness/multiseed_calibration.json`.

- Both variants use exactly the same calibration group split per seed, with
  distinct `video_id`s between temporary fit and calibration subsets.
- A full-train model with **fixed** forest RNG seed 42 is trained once per
  variant/target. This holds its predictions and **average precision constant**
  across different calibration seeds.
- For each calibration seed, a temporary model fits only that seed's
  train-video fit subset; its operating-point threshold is chosen using
  the held-out **train-video calibration subset**.
- The final fixed model is then evaluated on the same validation videos at
  each train-calibrated threshold. Report descriptive range, median, mean,
  per-seed paired temporal-minus-static values and win counts.
- No candidate is cherry-picked by validation. **Do not choose the seed that
  gives the highest validation F1.** Report all prespecified seeds.

## What this experiment does not establish

It measures threshold sensitivity to group calibration selection, **not**
randomness from re-fitting the final model at multiple random seeds or sampling
new subjects. It does not provide a new independent validation set. Repeatedly
examining the same validation set when proposing changes can still overfit the
research direction. Source video-ID disjointness does **not** prove subject
disjointness. Pose extraction accepted only ~50% of eligible frames.

For a defensible next stage, investigate why landmark visibility/detection
fails, then predeclare a final experimental configuration and evaluate the
publisher's untouched test videos once. Any model deployment will also need
video-level alarm stability, runtime profiling and safety validation.
