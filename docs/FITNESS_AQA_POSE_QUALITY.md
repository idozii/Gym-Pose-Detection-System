# Fitness-AQA: diagnose ~50% pose acceptance before changing models

## Motivation: observed full train + validation extraction

The user's run of 1,379 original Fitness-AQA source train/val squat clips
accepted **5,293 / 10,470** eligible sampled frames: ~50.6% coverage.
The 10,470 eligible frames exclude 894 frames near annotation boundaries.
Overall, 200 of the 1,379 clips (163 train + 37 validation) provided no
usable pose row.

This is the largest open reliability issue for a real-time coach.
The existing extractor combines several failure types into a single
`pose_rejected` count. We should not lower landmark visibility thresholds
blindly and then claim a model improvement.

## New diagnostic pilot (no retraining, no full video extraction)

From the repository root, using the original local ZIP and JSON annotations:

```bash
git fetch origin
git switch --track origin/feat/fitness-aqa-pose-quality-audit

python -m scripts.audit_fitness_aqa_video_quality \
  --videos-per-split 45 \
  --sample-fps 2 \
  --seed 42 \
  --min-visibility 0.5
```

This re-opens **up to 45 selected train clips and 45 selected val clips**
(one at a time from ZIP, with bounded temporary file storage).
It does not open test clips, create a new training dataset, save any
original frame image, or change the existing pose-feature CSVs.

Selection is **stratified by presence of publisher annotations** for
inward-knee, forward-only, or neither named error. It deliberately
increases coverage of rare inward-knee annotated videos, so its overall
acceptance fraction is **NOT** a population-weighted full-dataset estimate.
Its purpose is to identify rejection mechanisms, not estimate prevalence.

## Outputs (all ignored by Git)

```text
artifacts/fitness_aqa_pose_quality/
├── pose_quality_report.json
├── quality_train_samples.csv
└── quality_val_samples.csv
```

The report contains:
- first-rejection reason counts: no pose, low critical-joint visibility,
  degenerate shoulder width, invalid/nonfinite data, and accepted;
- which of the eight critical joints failed the current quality gate
  (left/right shoulder, hip, knee, ankle), counting joint frequency;
- acceptance among frames with/without each **named error annotation**
  at sampled timestamps, with annotation-boundary frames excluded;
- acceptance by sampled video-annotation stratum;
- **counterfactual quality gate coverage** at visibility cutoffs
  0.25, 0.35, 0.50 and 0.65, *without changing any training pipeline*;
- video failures and selected/eligible sample counts.

A frame can have multiple low-visibility joints: their counts must not be
added as mutually exclusive causes. No-pose detection and geometric invalidity
are mutually exclusive *first* rejection types. The CSV lists timestamp
and reason per sampled frame so that unusual videos can be inspected locally
without sharing private video content.

## Research decisions after reviewing report

1. If most rejection is **no pose**: look into pose detector crop/lighting,
   scale, camera motion and MediaPipe version/model complexity.
2. If most rejection is **critical-joint visibility**: inspect ankle/knee/hip
   joint distributions, viewpoint dependence and detection quality.
3. If a lower gate would raise coverage: run a controlled sensitivity
   experiment retaining the *same publisher splits*. Higher coverage alone
   is not necessarily better; unreliable joints may harm performance.
4. If acceptance depends heavily on error annotation: document selection
   bias explicitly and investigate before final model selection.

Do not touch the publisher's test split yet. No clinically valid technique
or injury-risk claims can be drawn from the pose-quality diagnostic.
