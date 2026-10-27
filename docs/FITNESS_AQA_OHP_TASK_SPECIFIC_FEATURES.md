# Milestone 13C - two task-specific OHP pose feature pipelines

## Motivation from the owner's real-data OHP coverage audit (PR #17)

Baseline OHP extraction at all 12 critical landmarks visibility >=0.50
accepted 2,090 / 13,496 train (15.49%) and 419 / 2,943 validation
(14.24%). The original 111-feature OHP format requires reliable elbow,
wrist, knee and ankle coordinates *simultaneously*.

In the annotation-stratified follow-up audit (48 train and 48 val clips),
we observed:

| Candidate | Train accepted / 387 | Val accepted / 411 | Train elbow-positive / 40 | Val elbow-positive / 52 | Train knee-positive / 33 | Val knee-positive / 35 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All 12 >=0.50 | 55 | 84 | 2 | 6 | 4 | 14 |
| All 12 >=0.35 | 81 | 105 | 3 | 10 | 6 | 17 |
| Upper 8 >=0.50 | 142 | 139 | 6 | 8 | 9 | 15 |
| Lower 8 >=0.50 | 112 | 181 | 9 | 21 | 10 | 21 |

**Upper 8** is relevant to learning **elbow** errors; the last two
knee columns in that row do NOT imply it is a reliable knee detector.
**Lower 8** is relevant to learning **knee** errors; its elbow-positive
count does NOT imply it can reliably predict elbow errors. A feature
must be extracted from a reliable joint *used by the model*.

The audit was deliberately stratified, NOT representative of all OHP
videos. Therefore do not extrapolate pilot percentages to full-data
coverage. The following new extraction is the full-population check.

## Two genuinely separate OHP feature schemas (30 numeric inputs each)

`ohp-elbows-upper8-30-v1`: 3 normalized coordinates for each of
shoulders (11,12), elbows (13,14), wrists (15,16) and hips (23,24),
plus left/right elbow angles, left/right normalized wrist height from
shoulder and normalized horizontal elbow/wrist widths.

`ohp-knees-lower8-30-v1`: 3 normalized coordinates for each of
shoulders (11,12), hips (23,24), knees (25,26), ankles (27,28), plus
left/right knee angles, left/right hip angles, camera-relative trunk
inclination and hip/shoulder width ratio.

Each task requires visibility >=0.50 for its own 8 selected joints only.
All engineered feature coordinates use ONLY required landmarks.
Unreliable irrelevant joints never enter the 30-element feature vector.

**Important:** Both are new, versioned schemas and are **not
interchangeable** with the historical 111-feature OHP RF. The
normalization and geometry are camera-relative and not clinically
validated; joint positions themselves can still be noisy.

## Run pilot using your existing labeled OHP ZIP

```bash
git fetch origin
git switch feat/m13-ohp-task-specific-pose-features 2>/dev/null || \
  git switch --track origin/feat/m13-ohp-task-specific-pose-features
git pull --ff-only
python -m pytest -q

python -m scripts.extract_fitness_aqa_ohp_tasks \
  --split train-val \
  --max-videos-per-split 8 \
  --sample-fps 2 \
  --boundary-margin 0.1 \
  --min-visibility 0.5
```

If the pilot has no decoder or feature schema failures, run:

```bash
python -m scripts.extract_fitness_aqa_ohp_tasks \
  --split train-val \
  --max-videos-per-split 0 \
  --sample-fps 2 \
  --boundary-margin 0.1 \
  --min-visibility 0.5
```

The full run processes exactly the original publisher TRAIN 1,582
and VAL 339 clips (never the 339 TEST clips). It processes each decoded
source frame **once**, and independently decides whether to emit one,
both or neither error-task training row. Error positives/negatives and
counts of videos with accepted and positive frames are reported.

Original media ZIP, annotations, 111-feature OHP CSVs and Squat
CSV/models stay unmodified. The new output directory is:

```text
data/features/fitness_aqa_ohp_tasks/
├── fitness_aqa_ohp_error_elbows_train_pose.csv
├── fitness_aqa_ohp_error_elbows_val_pose.csv
├── fitness_aqa_ohp_error_knees_train_pose.csv
├── fitness_aqa_ohp_error_knees_val_pose.csv
└── fitness_aqa_ohp_task_extraction_summary.json
```

The full run overwrites *only* the task CSVs created by an earlier
pilot in this new directory. Use `--output-dir` if you want to preserve
pilot data separately.

## Interpretation and next step

Each task's CSV has task-specific accepted frame IDs. Report both:
- accepted and positive frame counts, plus positive **video IDs**;
- pose acceptance as a fraction of annotation-eligible sampled frames.

A frame with `error_elbows == 0` means no **elbow** interval was
annotated at this timestamp; it is NOT a globally correct OHP lift.
Frame-level correlations are strong: 20 positive frames from 1 source
video are not 20 independent positives.

The next research step is a grouped train-only calibration, static RF
vs causal temporal RF benchmark **within each task schema**, evaluate
using original publisher validation (with clip-cluster bootstrap).
Do NOT compare raw AP/Macro-F1 from different accepted frame
populations as direct evidence of model superiority. Include a matched
sample-ID report to compare against the old 111-feature OHP extraction,
and exclude the publisher TEST split until the protocol is frozen.

This extraction does NOT train models and cannot report classification
accuracy by itself.
