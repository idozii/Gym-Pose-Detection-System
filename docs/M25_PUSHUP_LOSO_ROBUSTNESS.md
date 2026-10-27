# M25 - Push-up subject-heldout robustness and uncertainty

## Observed M24 results from owner's Ubuntu (27 October 2026)

M24 completed without reading original M20 VAL/TEST participants:

- Frozen original M20 TRAIN: **14 Push-up participants**.
- Full-body admitted **69 clips**; upper-body admitted **83**,
  recovering **14 additional front Push-up clips**.
- All three RFs were scored on the same 69 common clips with
  **5-fold GroupKFold** over those 14 original TRAIN participants.
- Every model used 200 trees, fixed seed 42 and identical M23
  masked 64-feature schema.

| M24 model | Balanced accuracy | F1 bad | Recall bad | AP bad |
| --- | ---: | ---: | ---: | ---: |
| `full_paired` | **56.22%** | 44.44% | 35.29% | **65.70%** |
| `upper_paired` | 47.69% | 41.94% | 38.24% | 55.14% |
| `upper_expanded` | 49.24% | **47.76%** | **47.06%** | 61.32% |

The extra recovered-only upper-expanded heldout TRAIN clips have
**9/14 correct predictions**, with 64.29% balanced accuracy and
61.54% F1 bad. However, this is a tiny **UNPAIRED** subset,
and the full gate abstains on these clips. It is not evidence of an
overall classifier advantage.

The full-paired reference had 8 false-positive `bad` predictions
on 35 `good` clips and missed 22/34 `bad` clips. Upper-expanded
raised false positives to 17 and reduced missed `bad` clips
to 18. Better `bad` recall was bought at a significant specificity
cost. None of the approaches currently supports a trustworthy
fitness-form claim on the public website.

## Research question

**Is the apparent difference between the model variants stable when
each ORIGINAL TRAIN participant is excluded in turn?**

M25 adds a robust *subject-heldout sensitivity check*:

1. Reconstruct the frozen M20 TRAIN set, verify all M23 video/label
   metadata, and restrict analysis to the same paired 69 clip IDs.
2. Run **Leave-One-Subject-Out (LOSO)** (14 folds) for each of
   the same three 200-tree RF approaches, with no participant's
   side, diagonal or front clips available to that person's fold
   training.
3. Compute pooled LOSO Balanced Accuracy, F1, recall, precision,
   average precision and confusion matrices, including per-view
   group breakdowns.
4. Evaluate the extra upper-only heldout clips separately, never
   blend them into paired common metrics.
5. Resample the 14 participants **with replacement 1,000 times**,
   on the **fixed out-of-fold predictions**, using identical
   resampled participant IDs in the full reference and two upper
   variants. Report paired differences in Balanced Accuracy and
   bad-label recall with approximate percentile intervals.

This cluster bootstrap does **not refit models for every bootstrap
draw**. Its uncertainty interval reflects subject sampling of
fixed LOSO predictions and is descriptive, not a calibrated external
prediction or formal proof of superiority. The number of source
participants is small; wide or mixed-sign intervals are expected.

## Execute on Ubuntu

The source dataset ZIP is **not needed** for this step.

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

python -m scripts.m25_pushup_loso_robustness
```

Outputs only Git-ignored aggregate JSON:

```text
data/processed/m25_pushup_loso_robustness.json
```

Print key results:

```bash
python - <<'PY'
import json

p = 'data/processed/m25_pushup_loso_robustness.json'
with open(p) as f:
    r = json.load(f)
print('Original TRAIN Push-up participants:', r['original_train_subjects_in_pushup'])
print('Paired clips:', r['paired_common_clips'])
print('Upper-only clips:', r['upper_only_recovered_clips'])
for name, item in r['model_scores_on_identical_clips'].items():
    x = item['paired_common_loso']
    print(name, 'BA:', round(x['balanced_accuracy'], 4),
          'F1 bad:', round(x['f1_bad'], 4),
          'Recall bad:', round(x['recall_bad'], 4),
          'n:', x['n'])
for name, metrics in r['cluster_bootstrap']['paired_delta_vs_full'].items():
    print(name, 'vs full:')
    for key, interval in metrics.items():
        print(' ', key, interval)
print('VAL/TEST scored:', r['validation_or_test_subjects_scored'])
PY
```

If output already exists, do not overwrite. Inspect it or use
a separate `--output data/processed/m25_second_run.json` for an
explicit reproducibility comparison.

## Interpretation and next scientific decision

- If the `upper_expanded` BA difference stays negative over
  LOSO and bootstrap, **do not advertise the upper gate as a
  quality-accuracy improvement**; it should remain a coverage
  mechanism with abstention/calibration requirements.
- If results change dramatically by participant, the owner should
  focus on participant and viewpoint generalization before
  tuning more classifiers. RF tree count alone is unlikely
  to solve weak video-quality labels.
- Consider investigating temporal exercise features, visibility
  masking effects, camera-specific calibration and per-subject
  normalization **on TRAIN only**.
- Once the candidate is fixed, acquire an independent validation
  result using the original M20 frozen VAL participants and
  compare on the same source cohort, reporting accepted/rejected
  proportions and both conditional accuracy and end-to-end
  coverage. Keep final TEST untouched until final protocol.

M25 does not train/publish a production model, deploy Push-up to
the existing single-frame API, upload private video or change paid
hosting. The current hosted six detectors continue to represent
original Fitness-AQA source annotation research only.
