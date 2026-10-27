# M31 - Bicep Curl TRAIN-only paired subject robustness

## Real M30 findings (owner's Ubuntu)

The owner completed M30 with 15 original TRAIN subjects,
90 Bicep Curl videos and a fixed 300-tree RF for each
of the four feature families. No original M20 VAL or
final TEST data were scored by M30.

| Feature group | Group5 Balanced Accuracy | LOSO Balanced Accuracy | LOSO bad recall | LOSO AP bad |
| --- | ---: | ---: | ---: | ---: |
| `baseline_264` | **60.00%** | **60.00%** | **62.22%** | **60.37%** |
| `kinematic_276` | 57.78% | 56.67% | 55.56% | 58.28% |
| `tracking_proxy_268` | 61.11% | 58.89% | 60.00% | 59.52% |
| `temporal_280` | **62.22%** | 57.78% | 55.56% | 56.87% |

Notes:

- The full 280-feature model improves Group5 BA by **+2.22
  percentage points**, but loses **-2.22 points** versus baseline
  in LOSO, and bad recall falls **6.66 points** in LOSO.
- The tracking proxy improves Group5 by **+1.11 points**, but
  falls **-1.11 points** on LOSO. This does not establish that
  it was exploiting camera/viewpoint artifacts, only that
  the effect was not stable across evaluation strategies.
- The interpretable 12 kinematic descriptors alone underperform
  baseline in both methods, so they have not yet helped this
  subject-level Bicep Curl task.
- The original `baseline_264` is the most stable *aggregate*
  choice across the two TRAIN cross-validation strategies.
  Equal aggregate scores **do not imply identical video-level
  predictions** between Group5 and LOSO.

The source labels remain the publisher's **whole-video
good/bad** labels. This is NOT an elbow-fault classifier,
physical-therapy quality assessment or injury diagnosis.

## M31 research question

Do the per-person differences between the original 264-feature
reference and the three augmented feature families remain in
the same direction across held-out subjects, and are they
larger than subject sampling variability?

M31 reruns the **same M30 settings** on the **same frozen
original TRAIN clip IDs**, generating private in-memory
out-of-fold predictions for both original GroupKFold(5) and
Leave-One-Subject-Out (LOSO). It computes:

- Exact reproduction of previous owner's M30 metrics
  for **all four feature sets, both strategies**. If the old
  report exists and any BA, F1 bad, recall bad, AP bad,
  subject count, video count or fold count differs at
  1e-12 precision, M31 **refuses to write results**.
- Paired **1,000-resample participant-cluster bootstrap**
  intervals for alternative minus `baseline_264` Balanced
  Accuracy and bad-label recall, using identical sampled
  original TRAIN subjects/video IDs for both methods.
- Anonymous counts of original TRAIN participants on whom
  each candidate gives more, fewer or the same number of
  correct video-level classifications than the baseline.
- Aggregate per-camera front, diagonal and side results.
  No per-person/video predictions, consent information,
  source filenames, people IDs or publisher source video
  frames are saved in the report.

**Critical limitation:** resampling fixed out-of-fold
predictions does not refit Random Forests and does not
account for all training/model-selection uncertainty.
With 15 participants, percentile intervals are a
descriptive diagnostic, **not a formal test of superiority
on independent data**. Since this work follows previous
validation inspection, all results remain exploratory.

## Run on Ubuntu (no source ZIP or MediaPipe needed)

The following files should already exist from M20 and M30:

```text
data/features/m20_gym7/m20_clip_features.csv
data/features/m30_bicep_curl_train_motion_v1.csv
data/external/mendeley_gym7/m20_manifest.csv
data/processed/m30_curl_train_cv.json
```

Commands:

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate
python -m scripts.m31_curl_paired_stability
```

New Git-ignored output:

```text
data/processed/m31_curl_paired_subject_stability.json
```

Show the important statistics:

```bash
python - <<'PY'
import json
p="data/processed/m31_curl_paired_subject_stability.json"
with open(p) as f: r=json.load(f)
print("M30 reproduction:",r["m30_reproducibility"])
print("TRAIN people:",r["train_subjects"],"paired videos:",r["paired_train_clips"])
for cv, models in r["strategies"].items():
    print("\n",cv.upper(),"folds:",models["folds"])
    for family, metrics in models["models"].items():
        x=metrics["all_paired"]
        print(family,"BA:",round(x["balanced_accuracy"],4),
              "bad Recall:",round(x["recall_bad"],4))
    print("Paired subject intervals versus baseline:")
    for family, deltas in r["cluster_bootstrap"][cv][
            "deltas_vs_baseline_264"].items():
        print(family,"Delta BA:",deltas["balanced_accuracy"],
              "Delta bad recall:",deltas["recall_bad"])
    print("Participant direction counts:",
          models["subjects_with_better_or_worse_accuracy_vs_baseline"])
print("Original VAL/TEST scored:",r["original_VAL_or_TEST_scored"])
PY
```

Do NOT re-run the original M30 extract task: M31 uses only
existing feature CSV files and never decodes a video.
It will also refuse to overwrite an existing M31 JSON report.

## Research and deployment decisions

If candidate delta BA intervals cross zero, treat the
increased feature count as unproven; maintain the original
264-feature model as Bicep Curl research baseline.
Do not select a new model based on one fold/participant
or keep iterating against the repeatedly consulted
original M20 VAL set. Keep final original M20 TEST closed
until a documented final model and protocol are frozen.

An eventual Bicep Curl web feature could offer **local pose
geometry exploration**, similar to the currently available
Push-up mode, but publisher good/bad model predictions would
require a separate validated and versioned *temporal* browser
pipeline with cross-estimator feature parity. **M31 does not
deploy another quality model.**

The already merged Push-up web local preview and six
original Fitness-AQA source-label models are unchanged.
