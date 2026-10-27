# Milestone 15B - OHP source error feature adapter is integrated into the offline registry

## Why the owner's M15A output initially reported missing OHP schemas

The owner ran the M15A local smoke tests and received:

- **156 passed, 4 warnings**.
- `squat/error_knees_forward`, `squat/error_knees_inward`:
  `not_trained_as_source_error_detector` (intentional; the old notebook's
  correct/incorrect model is NOT either original Squat error detector).
- `ohp/error_elbows`, `ohp/error_knees`:
  `feature_schema_not_integrated` (registry was installed but
  `src/gym_pose/ohp_task_features.py` was absent on `main`).
- `barbell_row/torso_angle`, `barbell_row/lumbar`:
  `loaded_experimental_only` (both locally trusted M14D model
  bundles passed schema/provenance checks).

PR #18 retained the original OHP 30-feature training adapter on its
unmerged research branch; PR #19 similarly holds reproducible training
scripts. At the time this fix was created those PRs were still open and
the original OHP source code was absent on `main` although the
user-produced static joblib artifacts already existed locally.

## What this fix changes

- Ports the **exact** previously validated OHP task source geometry from
  PR #18 into `src/gym_pose/ohp_task_features.py`:
  - `error_elbows`: `ohp-elbows-upper8-30-v1`, shoulders, elbows,
    wrists, hips, 30 features, visibility >=.50 for the 8 used joints.
  - `error_knees`: `ohp-knees-lower8-30-v1`, shoulders, hips, knees,
    ankles, 30 features, identical visibility quality principle.
- Makes both schemas direct imports of the merged
  `gym_pose.model_registry` instead of silent, optional imports.
  Registry still checks exact exercise, target, schema version, 30 feature
  names/order, model binary classes, fitted input count, variant
  `static`, source TRAIN-only threshold and untouched publisher TEST
  metadata.
- Retains **separate** existing Barbell Row torso4 20-feature models and
  leaves Squat source-label slots **unavailable** until separately
  proven source-error artifacts are integrated.
- Adds tests for two trusted synthetic OHP bundles, verified source
  feature vectors, low-visibility abstention, bogus 111-feature/casual
  temporal artifacts, invalid thresholds, missing models, and
  never substituting an unrelated exercise model.

**No extraction, training, re-download, publisher TEST evaluation,
model accuracy claims, webcam integration, or changed thresholds.**

## Test on your existing machine

```bash
git fetch origin
git switch fix/m15b-ohp-source-feature-adapter 2>/dev/null || \
  git switch --track origin/fix/m15b-ohp-source-feature-adapter
git pull --ff-only

python -m pytest -q

# First command inspects status without loading any pickle:
python -m scripts.check_fitness_aqa_model_registry

# Explicit opt-in: only your OWN trained joblib artifacts:
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts
```

If the user's existing four model bundles from M13D and M14D have not
been moved, expected OHP/Row output is:

```json
{
  "ohp/error_elbows": "loaded_experimental_only",
  "ohp/error_knees": "loaded_experimental_only",
  "barbell_row/torso_angle": "loaded_experimental_only",
  "barbell_row/lumbar": "loaded_experimental_only"
}
```

Squat still returns `not_trained_as_source_error_detector` for its
two named knee-error slots. If an OHP model is missing, expect
`model_artifact_missing` instead; if it has an incompatible schema
or variant, loading deliberately fails with a provenance/schema error
rather than pretending the result is a valid source-error score.

Only joblib files saved by trusted local code should be loaded: joblib
deserialization executes pickle bytecode and is not sandboxed.
Local files are ignored by Git and remain on disk when switching
branches. Do **not** run `git clean -fdx` and do not overwrite the
locally modified historical notebook.

## Research and integration limitations

The above checks establish **software contract and source-label
format compatibility**, not accuracy. Both OHP and Barbell Row models
have low real-world-relevant recall or precision in their original
exploratory publisher validation research; neither is a validated
medical/injury-prevention form detector. `source_label_flag` refers
to a specific publisher annotation, not a safe lift.

M15B does not merge the entire PR #18/#19 research history by itself.
Before final release, original OHP training/reproducibility scripts and
overlapping research PRs should be reconciled against updated `main`.
Do not merge duplicate feature files blindly. The next major missing
adapter is the two real Squat **source knee error** model contracts,
not a renamed generic correct/incorrect notebook classifier.
