# Milestone 13A - Fitness-AQA OHP pose extraction (publisher train/val only)

Milestone 12 **source integrity already passed on the owner's computer**:

- Squat: 1,623/1,623 official video IDs present.
- OHP: 2,260/2,260 original clip IDs present, 1,582 train + 339 validation
  + 339 test; 107 extra clips in archive intentionally excluded.
- Barbell Row: 18,201/18,201 assigned image IDs present for the union of
  torso-angle and lumbar tasks; no cross-task split-ID conflict reported.

M12 scanned ZIP metadata but **did not** decode videos, extract poses,
benchmark models or evaluate the test split.

## What M13A adds

- Exercise-specific OHP feature schema `ohp-pose-111-v1`: original
  103 MediaPipe Pose coordinates/geometry features (for continuity)
  **plus eight** OHP-specific joint geometry features: left/right elbow
  angles, individual knee angles, vertical wrist-to-shoulder differences,
  horizontal elbow width and wrist width normalized to shoulder distance.
- Upper limb visibility gate for elbows and wrists (indices 13-16)
  **in addition to** the original critical joints (shoulders, hips,
  knees and ankles); 0.50 by default. Coverage is explicitly reported
  rather than assumed sufficient.
- Reuses the existing timestamp-based annotation union and 0.1-second
  boundary exclusion, including the authentic publisher OHP out-of-order
  time windows and slight elbow overlap (without editing source JSON).
- Opens a single original OHP video at a time, stores no raw frames;
  discards incomplete video CSV rows on clip failures; logs causes.
- Preserves source `video_id`, `split`, sample `timestamp_s`,
  `frame_index`, `source_fps`, and **two distinct** named binary
  targets `error_elbows` and `error_knees`.
- Never processes publisher TEST video. The source ZIP inspector sees
  ZIP central-directory filenames and labels for integrity only.

## Pilot first (already downloaded source media)

```bash
git fetch origin
git switch --track origin/feat/m13-ohp-pose-extraction
python -m pip install -e '.[camera,dev]'
python -m pytest -q

python -m scripts.extract_fitness_aqa_ohp \
  --split train-val --max-videos-per-split 8 \
  --sample-fps 2 --boundary-margin 0.1 --min-visibility 0.5
```

Output directory is Git-ignored:

```text
data/features/fitness_aqa_ohp/
├── fitness_aqa_ohp_train_pose.csv
├── fitness_aqa_ohp_val_pose.csv
└── fitness_aqa_ohp_extraction_summary.json
```

The pilot is a tiny smoke test only. It may have zero retained
positive-class errors or videos depending on alphabetical clip selection.
Do not train or report model performance from a tiny pilot.

## After pilot succeeds

Use the **same** command with `--max-videos-per-split 0` to extract the
complete 1,582 source TRAIN and 339 VAL clips (NOT 339 source TEST clips):

```bash
python -m scripts.extract_fitness_aqa_ohp \
  --split train-val --max-videos-per-split 0 \
  --sample-fps 2 --boundary-margin 0.1 --min-visibility 0.5
```

**Important:** the full extraction reuses the default output directory
and therefore overwrites the *pilot* train/val CSVs; it never overwrites
squat CSVs, OHP archive/annotations, test data or Git-tracked files.
Use `--output-dir` if you wish to retain pilot CSVs separately.

Inspect the summary: selected videos, successfully decoded clips,
rejected/accepted poses, annotation-boundary exclusions, per-target
accepted positive sample counts, and accepted-frame coverage.
A low pose coverage or too few positive error frames should
block model-fit claims until further checks.

## Subsequent M13B model benchmark (not part of this PR)

Once you share the **full run's JSON**, build:
- OHP train/val schema validator, source-video leakage checks;
- OHP-specific static RF and causal temporal feature ablation;
- train-video-group threshold calibration, no validation threshold fitting;
- AP / Macro-F1 / recall / precision and video-cluster uncertainty;
- threshold stability and pose coverage reporting;
- test set locked until experiment protocol is frozen.

**No subject identity is known:** publisher clip-disjoint is NOT a
subject-independent benchmark. A zero for one annotation is not
globally correct OHP technique. No injury-prevention claim.
