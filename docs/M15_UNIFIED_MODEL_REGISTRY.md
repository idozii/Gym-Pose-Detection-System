# Milestone 15A - offline multi-exercise model registry / source-label inference contract

## Stage and evidence

M14D Barbell Row was run on **real** publisher TRAIN and source VALIDATION
with **140 tests passing** and the original publisher TEST unopened.

| Task | Pose coverage train/val | Validation source-label AP | Source-label prevalence reference | Calibration threshold | Macro F1 | Positive precision/recall |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Row `torso_angle` | 70.14% / 69.77% | **.4460** | .1118 | .30 | .6908 | .4408 / .4650 |
| Row `lumbar` | 70.20% / 70.86% | **.5056** | .1591 | .50 | .6900 | .6133 / .3665 |

These are **exploratory source validation** results, not test or
cross-athlete scores. In particular, the Row torso4 shoulder+hip
skeleton cannot directly measure lumbar spinal curvature. Row recall
of 46.5% / 36.7% remains too weak for confident coaching alerts.

Earlier OHP results: task-specific static vs causal temporal RF
explored elbow and knee source error annotations. Across five seeds,
knee temporal exceeded static AP/F1/recall in all five, but the
original validation set was reused repeatedly and has limited
positive source clips. **This does not justify real-time advice.**

## Six planned named publisher-source error targets

| Exercise | Source target | Registry status today |
| --- | --- | --- |
| Squat | `error_knees_forward` | Reserved only; distinct source error model has NOT been loaded/registered |
| Squat | `error_knees_inward` | Reserved only; distinct source error model has NOT been loaded/registered |
| Overhead Press | `error_elbows` | Supports the research static 30-feature artifact **after** OHP source feature code from PR #18/#19 is merged |
| Overhead Press | `error_knees` | Same |
| Barbell Row | `torso_angle` | Supports local torso4 static 20-feature artifact from M14D |
| Barbell Row | `lumbar` | Supports local torso4 static 20-feature artifact from M14D |

**Do NOT confuse the legacy Squat notebook correct/incorrect binary
classifier with either publisher Squat knee error label.** The schema and
label semantics differ and this registry deliberately refuses that
substitution.

## Runtime contract

`gym_pose.model_registry.ModelRegistry`:
- Takes only locally trusted `.joblib` bundles. Joblib uses pickle and
  deserialization can execute code; explicitly opt in with
  `trust_local_pickle_files=True`, only for your **own** saved artifacts.
  No URL fetching, model registry uploads, or publisher test decoding.
- Verifies per-artifact exercise, original task name, schema version,
  exact model input feature order, fitted model numeric classes `[0,1]`,
  feature count, train-calibration threshold, and source TEST untouched
  provenance assertions before making the model available.
- A missing artifact/feature schema returns *abstain*, not an arbitrary
  fallback, negative classification, or 'good form'.
- A failed per-task MediaPipe visibility gate, missing or nonfinite
  source features, incorrect feature count/schema or invalid probability
  returns *abstain*.
- A successful score is explicitly
  `experimental_source_label_score` with
  `error_probability`, train-selected `threshold`,
  and `source_label_flag`. These are neither calibration-validated
  injury risks nor reliable coaching instructions.
- Only **static** OHP 30-feature input is supported in M15A. Causal
  temporal 42-feature models require a separate timestamp/history
  manager and are intentionally not silently accepted here.
- Keeps Row shoulder+hip 20-feature schema distinct from OHP 30-
  feature variants. No cross-exercise schema coercion.

## Local availability check (does not load any pickle by default)

```bash
git fetch origin
git switch feat/m15a-unified-model-registry 2>/dev/null || \
  git switch --track origin/feat/m15a-unified-model-registry
git pull --ff-only
python -m pytest -q

# Inspect the six expected error slots without deserializing artifacts:
python -m scripts.check_fitness_aqa_model_registry

# Only if these are the user's OWN locally trained joblib files:
python -m scripts.check_fitness_aqa_model_registry --trust-local-artifacts
```

Expected locally saved artifact names:
```text
artifacts/
├── fitness_aqa_ohp_task_rf/
│   ├── error_elbows_static.joblib
│   └── error_knees_static.joblib
└── fitness_aqa_barbell_row_torso4_rf/
    ├── barbell_row_torso_angle_torso4_static_rf.joblib
    └── barbell_row_lumbar_torso4_static_rf.joblib
```

**Current GitHub dependency:** PR #18 (OHP task features) and PR #19
(OHP trained benchmark) were still open when M15A began. While those
source modules are absent on `main`, the registry marks OHP slots
`feature_schema_not_integrated`, not trained or ready. Squat source
error slots are also deliberately not configured.

The registry does not ship or download joblib artifacts; they are
ignored local outputs on the owner's machine. Switching Git branches
must preserve the local artifacts directories.

## Next steps

- M15B: merge/integrate OHP source adapters and model contracts,
  separately package the Squat source-error detector artifacts with
  verified model metadata, exercise-aware inference tests and explicit
  abstention coverage.
- M16: only after the models are correctly registered, build
  source-video-style temporal state and an optional real-time UI.
  Weak/high-FPR detectors must not be presented as reliable warnings.
- M17: performance and generalization evaluation under a frozen
  protocol with clear model and validation limitations.

The endpoint here is a **safe-to-inspect software contract**, not
certification of physical exercise form, medical guidance, or a
production-ready coach.
