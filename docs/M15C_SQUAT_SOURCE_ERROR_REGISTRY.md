# Milestone 15C — genuine Fitness-AQA Squat named-error RFs, six-slot registry

## Owner-confirmed M15B status

The owner ran M15B followed by the model registry isolation fix:
**167 tests passed, 4 preexisting sklearn single-label warnings**.
All four OHP/Barbell Row experimental local model bundles loaded
successfully, with original publisher TEST untouched.
The two Squat source error slots still reported
`not_trained_as_source_error_detector`.

The existing project ALREADY has two independent Squat **source
annotation** labels in the train/val pose CSVs:
`error_knees_forward` and `error_knees_inward`. Those are original
publisher error-time-interval targets, NOT the unrelated notebook
`correct/incorrect` classification.

The earlier `scripts/benchmark_fitness_aqa_errors.py` did train
target-specific RFs but wrote old bundles with `decision_threshold=.5`,
without the source-TRAIN-only group calibration or strict metadata
`source_train_only` and `publisher_test_untouched` required by the
new merged registry. Merely renaming those old artifacts or inventing
train-calibration provenance would be scientifically misleading.

## M15C implementation

- `scripts/train_fitness_aqa_squat_source_errors.py` reads the
  already-extracted **original publisher TRAIN and VAL** Squat CSVs
  and full extraction manifest only; does NOT open video ZIP or TEST.
- Enforces `squat-103-v1` exact 103 features, metadata columns,
  original source split type, 2 FPS, 0.1s boundary margin,
  visibility>=0.5, complete original publisher TRAIN 1,136/VAL 243
  videos, zero video failures, source labels and count agreement.
  Refuses partial pilot or extraction manifests with TEST processed.
- For each original target independently, a temporary RF is trained
  on a disjoint group of source TRAIN clip IDs and its operating
  threshold is selected on held-out **TRAIN video groups only**.
  RF parameters: 300 trees, `min_samples_leaf=2`,
  `class_weight='balanced'`, seed 42. A new final RF is then fit
  to all original source TRAIN accepted frames only.
- Original publisher VAL is evaluated at both fixed 0.5 and the
  train-selected threshold; reports AP vs named-error prevalence,
  Macro-F1, precision, recall, FPR and the number of positive source
  videos. VAL is for exploratory reporting only, and has already
  been examined repeatedly during the project.
- Creates versioned and provenance-checked artifacts:
  `rf_error_knees_forward_static_source.joblib` and
  `rf_error_knees_inward_static_source.joblib`.
  These are separate from old generic
  `models/squat_form_rf.joblib`, and do not change
  existing Squat/Squat temporal notebook models.
- Extends `gym_pose.model_registry.ModelRegistry` to include actual
  Squat source-error contracts. It explicitly checks
  `annotation_semantics='publisher_named_error_interval_NOT_correctness'`,
  target identity, fitted 103-column schema, `static` variant,
  binary [0,1] classes, TRAIN-only threshold provenance and no TEST.
  A missing/incompatible model/pose **abstains**. No guess, coaching
  or global `correct` label is emitted.

## Run on your Ubuntu machine

PR #28 is a new feature branch from up-to-date main, independent of
the historical overlapping PR #18/#19.

```bash
git fetch origin
git switch feat/m15c-squat-source-error-registry 2>/dev/null || \
  git switch --track origin/feat/m15c-squat-source-error-registry
git pull --ff-only
python -m pytest -q

# Train only if your original full publisher Squat TRAIN+VAL CSVs
# and extraction manifest are already available in data/features/fitness_aqa
python -m scripts.train_fitness_aqa_squat_source_errors \
  --features-dir data/features/fitness_aqa \
  --output-dir artifacts/fitness_aqa_squat_source_rf \
  --seed 42 --trees 300

# Only deserialize artifacts made on your OWN trusted machine:
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts
```

The new local outputs, ignored by Git:

```text
artifacts/fitness_aqa_squat_source_rf/
├── rf_error_knees_forward_static_source.joblib
├── rf_error_knees_inward_static_source.joblib
└── validation_report.json
```

**Do not rerun Squat video pose extraction** unless the source feature
CSVs or full publisher extraction manifest are actually missing.
The new script rejects partial pilot CSVs, source manifest mismatch,
cross-train/validation clip overlap, and nonbinary labels rather than
silently making unverified research claims.

Only files under the new output directory are created/replaced.
The original 103-feature pose data, old generic correctness model,
local notebook modifications and publisher TEST stay untouched.

### Expected availability after the training succeeds

```text
squat/error_knees_forward        loaded_experimental_only
squat/error_knees_inward         loaded_experimental_only
ohp/error_elbows                 loaded_experimental_only
ohp/error_knees                  loaded_experimental_only
barbell_row/torso_angle          loaded_experimental_only
barbell_row/lumbar               loaded_experimental_only
```

This represents **six source-error model contracts**, NOT verified
independent generalization or any guarantee of reliable real-time
form detection.

## Interpretation and onward milestones

Squat publisher split is video-ID-disjoint but not established
subject-disjoint; adjacent accepted frames are correlated. Absent
named annotations are not globally correct technique. Pose coverage
is incomplete; probabilities may not be calibrated in the
statistical sense despite choosing an operating classification
threshold. This code makes no deployment-safety claims.

Upon owner-confirmed six-slot load, M15 inference integration can be
considered *contract-complete*. M16 can implement optional
real-time research UI while **withholding automatic high-confidence
injury/form warnings for underperforming model tasks**. M17 requires
release testing, frozen protocol and transparent quality limits.
