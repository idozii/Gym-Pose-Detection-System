# Fitness-AQA: knee-aware visibility gate feasibility, without reprocessing video

## What prompted this?

The user's **full-data** ablation compared pose visibility 0.50 and 0.35
on original publisher train+validation splits, retaining respectively:

| Source split | Gate 0.50 | Gate 0.35 |
| --- | ---: | ---: |
| Train accepted / eligible | 4,309 / 8,582 (50.2%) | 5,442 / 8,582 (63.4%) |
| Validation accepted / eligible | 984 / 1,888 (52.1%) | 1,207 / 1,888 (63.9%) |

The 0.35 gate admitted 1,133 more train and 223 more validation frames
with **zero change in the original 0.50 accepted frame features**.
It was not a uniformly better model: knee-inward common-frame macro-F1
fell 0.6277 -> 0.5996 and precision fell 0.25 -> 0.1892.
Knee-forward common-frame recall fell 0.8037 -> 0.7477 although
macro-F1 rose slightly. Therefore do NOT automatically lower the production
0.50 gate for every critical joint.

The earlier 90-video quality pilot identified low knee and ankle visibility
as the primary reason for rejection. It already emitted CSV rows with
failure-joint names and **minimum critical-joint visibility** at gate 0.50.

## New analysis: evaluate more selective admission rules offline

```bash
git fetch origin
git switch --track origin/feat/fitness-aqa-joint-selective-gate-diagnostic

python -m scripts.diagnose_fitness_aqa_joint_gates
```

This reads only:
- `artifacts/fitness_aqa_pose_quality/quality_train_samples.csv`
- `artifacts/fitness_aqa_pose_quality/quality_val_samples.csv`
- `artifacts/fitness_aqa_pose_quality/pose_quality_report.json`

No ZIP or camera processing, no original video, model training, model
loading or test access. Report:
`artifacts/fitness_aqa_joint_gates/counterfactual.json`.

### Four candidate gate categories, plus strict baseline

- `all_050`: require all eight critical joints to have visibility >=0.50.
- `ankles_035_other_050`: ankle joints may be >=0.35 while
  shoulders/hips/knees remain >=0.50.
- `knees_035_other_050`: knee joints may be >=0.35 while
  shoulders/hips/ankles remain >=0.50.
- `knees_ankles_035_other_050`: knees/ankles >=0.35;
  shoulders/hips remain >=0.50.
- `all_035`: all eight critical joints >=0.35.

Because the original QA recorded *every joint failing gate 0.50* and
the minimum across critical-joint visibility, this is an exact
**gate-admission counterfactual** on those **already sampled frames**,
with the other existing rejection reasons unchanged. Script verifies
the reconstructed `all_050` and `all_035` accepted counts against
the saved quality audit before reporting any candidate.

For each gate it reports overall acceptance, newly admitted frames,
number of videos with at least one accepted pose, and new positive/negative
annotation frames for both named error tasks.

### Interpretation

The 90 pilot clips were **annotation-stratified**, not a representative
sample of the whole dataset, so no projected training set size or expected
population-weighted coverage should be inferred from those rates.
Newly admitting an error-annotated frame does not guarantee a usable
landmark feature vector or a correct classifier prediction.

**Decision next:** If relaxing ankles only recovers meaningful data
while keeping knee visibility strict, consider an actual train/validation
re-extraction using an asymmetric gate. But first inspect the proportions
and error-label retention, then use the same clip-split, shared-frame
and fixed validation comparisons applied in PR #10.

Keep the source test split untouched until one configuration and one
evaluation protocol are frozen. 0 in a named error target is absence of
that specific annotation, not a certification of perfect squat form.
