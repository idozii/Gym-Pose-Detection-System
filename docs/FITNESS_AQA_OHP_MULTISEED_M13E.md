# M13E - OHP RF multi-seed stability (TRAIN video groups only)

This study is a follow-up to M13D's **owner-run** OHP static vs causal
temporal RF benchmark (PR #19). It probes whether the observed advantage
is stable across independent *random seeds* for model fitting and the
source-TRAIN video-group calibration partition. **The seeds are not
independent source datasets**, so a seed sign count is descriptive, not a
statistical significance test.

## Actual M13D evidence motivating this check

All runs below use the original Fitness-AQA OHP publisher
train/validation only; the publisher test split remains locked.

| Error target | Source val positives | Static AP | Temporal AP | Static macro F1 | Temporal macro F1 | Paired video-bootstrap AP-difference 95% |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Elbow | 25 accepted frames, 14 videos | .2858 | .3411 | .6941 | .6849 | [-.1124, +.2100] |
| Knee | 73 accepted frames, 43 videos | .1439 | .1957 | .5552 | .5973 | [+.0041, +.1229] |

For elbow, a temporal gain in AP is plausible, **but not established**
on the already-used validation set. Temporal recall improves 56% -> 68%
while precision falls 32.6% -> 28.3%, and FPR rises 3.37% -> 4.99%.
For knee, temporal improves recall 17.8% -> 24.7%, precision 16.7% -> 25%
and reduces FPR 6.40% -> 5.32%; absolute recall remains poor.

The original M13D implementation calibrates each model's operating
threshold from **one** publisher source-TRAIN video-group holdout.
The new script repeats the complete temporary fit -> threshold
calibration -> final fit -> publisher validation scoring over a
predeclared list of seeds. It uses the **same fit/calibration source
video IDs for both variants within each target and seed**.
The OHP elbow and knee tasks use separate feature schemas and
different accepted-source-frame populations, so cross-task performance
comparisons have different denominators and are not head-to-head.

## Run (no ZIP downloads, MediaPipe, or source TEST required)

PR #20 stacks on PR #19 (which stacks on PR #18); checkout the
branch directly if its parents have not yet merged.

```bash
git fetch origin

git switch feat/m13-ohp-multiseed-stability 2>/dev/null || \
  git switch --track origin/feat/m13-ohp-multiseed-stability

git pull --ff-only
python -m pytest -q

python -m scripts.assess_fitness_aqa_ohp_seed_stability \
  --features-dir data/features/fitness_aqa_ohp_tasks \
  --output artifacts/fitness_aqa_ohp_task_rf/seed_stability.json \
  --seeds 41,42,43,44,45 \
  --trees 150 \
  --max-gap-seconds 0.75
```

This runs 5 predeclared seeds on the **existing, full source TRAIN/VAL**
task CSVs and their manifest. It trains temporary Random Forests
in memory; it **does not overwrite** the four M13D `.joblib`
models or the M13D `comparison.json`. The lighter 150-tree
setting assesses sensitivity economically; M13D used 300 trees.
Different RF tree counts limit any exact seed-42 numerical equality
with M13D, so compare only variants **inside each robustness run**
and treat M13D's 300-tree model as the separate primary benchmark.

## Output: what to review

`artifacts/fitness_aqa_ohp_task_rf/seed_stability.json` includes:

- Per-task numbers of eligible/accepted source frames and annotation
  positive source videos.
- Each seed's **same video-group fit/calibration assignment** for static
  and causal temporal; only original train clips are used.
- Train-selected thresholds for static and temporal, and
  validation AP, macro-F1, recall, precision, FPR (plus fixed 0.5
  metrics) on identical validation frames within each task.
- Min, median, max, mean and across-seed sample SD for scores and
  thresholds.
- Counts of seeds with positive/negative/tied temporal-static AP,
  macro-F1, recall and precision differences.

These are descriptive sensitivity diagnostics. Repeated model
and validation analysis does not create a pristine held-out
benchmark. Do NOT optimize thresholds by picking a seed using
publisher validation. Do NOT interpret the fraction of positive
seeds as a p-value. Preserve the source test lock until a
protocol/model choice is frozen.

## Decision path

- If **elbow** consistently shows AP gain but declines at the train
  operating point, prioritize reducing false alerts and assess a
  clip-level temporal aggregation on a separate future evaluation
  source; don't deploy elbow warnings yet.
- If **knee** temporal improves ranking across seeds, it remains a
  useful research baseline but recall ~25% at the current
  operating threshold is insufficient for dependable coaching.
- Unless a concrete model-design blocker appears, the next feature
  expansion should be **Barbell Row**, respecting its per-image
  torso-angle/lumbar labels and task-specific publisher source splits.

No source TEST evaluation, medical interpretation, athlete independence
claim, or production accuracy claim is authorized by this study.
