# M22 - Controlled Push-up camera admission ablation (TRAIN only)

## Why this experiment

Owner M21 results for **15 rejected front Push-up clips** belonging
exclusively to M20 TRAIN subjects:

| M21 frame category | Count | Of 513 sampled frames |
| --- | ---: | ---: |
| `low_one_side_visibility` | 262 | 51.1% |
| `no_pose` | 227 | 44.2% |
| `accepted` | 24 | 4.7% |
| `low_torso_scale` | 0 | 0% |
| `invalid_landmarks` | 0 | 0% |

The rejected subjects include eight original publisher "bad" clips
(279 frames; 13 valid), and seven "good" clips (234 frames; 11 valid).
No rejected TRAIN-only clip reached the minimum four accepted frames.

The full Gym7 extraction had 266/294 accepted clips, with 28 total
rejected front Push-up clips (14 original good + 14 original bad).
These are **observations**, not a form-quality performance metric.

The same-side full-body visibility rule likely contributes heavily:
the M21 missing-joint counts were especially high for both ankles and
knees. **Each frame can be missing several joints**, so missing-joint
counts are not disjoint and must NOT be interpreted as percentages.

About 44% of M21 samples had no detected pose at all, which a gate
change alone cannot recover.

## M22 scientific question

On the **same sampled frames from the same 15 TRAIN-only rejected
video clips**, compare:

| Detector | Model complexity | Pose detection confidence |
| --- | ---: | ---: |
| baseline | 1 | 0.50 |
| heavy | 2 | 0.50 |
| lower_detection_threshold | 1 | 0.35 |

For every detector, compute two separate **admission policies**:

- **full_body**: original M20 requirement for one complete
  shoulder/elbow/wrist/hip/knee/ankle side at visibility 0.35 and
  nondegenerate shoulder/hip torso scale.
- **upper_body**: experimental requirement for one complete
  shoulder/elbow/wrist/hip side at visibility 0.35 and the same torso
  scale. **No knee or ankle visibility condition.**

This is a **paired diagnostic**, not a production feature schema and
not a qualified Push-up good/bad classifier. No predictions are
generated. No unobserved landmarks are imputed.

Metrics are frame acceptance count, failure reason by original good/bad
clip label, and clips reaching at least four valid frames. A camera
detector change can affect both whether a pose exists and landmark
visibility. The upper-body gate cannot recover an entirely missing pose.

## How to run on the owner's Ubuntu computer

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate
python -m scripts.m22_train_only_pose_ablation
```

The script reuses the **unchanged M20 subject split seed=42** and
requires the local Mendeley Exercise-wise ZIP, M20 reviewed manifest,
and original accepted feature CSV. It decodes only originally rejected
**front Push-up clips for TRAIN subjects**. It cannot open validation
or TEST participant videos and does not import any model bundle.

Output:

```text
data/processed/m22_front_gate_detector_ablation.json
```

The new 3-detector comparison may consume more CPU and memory than
M21 and initializes MediaPipe Pose three times. You can pilot the
first six TRAIN-rejected clips with `--limit 6` and then run the full
audit under a DIFFERENT `--output` path (never overwrite prior data).
Video files are streamed from the ZIP one by one, and only anonymized
aggregated counts are saved.

### Decision rule (no model training at this step)

1. Compare **baseline/full_body** M22 counts against M21 for these
   15 clips as a reproducibility sanity check. Both use the original
   frame sampling, threshold, torso normalization and complexity=1
   pose detector.
2. If baseline/upper_body gains enough accepted frames, verify that
   the accepted raw XYZ values and temporal coverage are suitable for
   a new versioned feature schema. Do not train existing 264-feature
   M20 models on those newly admitted clips without full retraining.
3. If many samples remain `no_pose`, compare heavy vs baseline
   at the **same gate** to isolate detector effects. The lower-detection-
   threshold mode is a separate exploratory variable.
4. Only after selecting one approach on TRAIN data, plan a new
   *predeclared* extraction for **all** clips with the same fixed
   subject IDs and splits. Evaluate coverage by camera, good/bad label,
   and participant. Independently assess classifier performance on
   VAL. Preserve the final TEST subjects and evaluation.
5. Do not interpret higher accepted-frame fraction as improved exercise
   form accuracy or injury-safety validity; the new feature schema
   needs fresh paired source/browser feature comparison and
   person-independent model validation before live deployment.

## Privacy and test integrity

- No video files, frame images, numeric landmark arrays, subject IDs,
  clip paths, or token values are persisted in the report.
- All inputs and results remain in Git-ignored `data/external/` and
  `data/processed/` paths.
- The new tool never modifies the existing M20 feature CSV or RFs.
- It never changes the six deployed original Fitness-AQA research
  detectors or the website.
- MediaPipe runtime warnings about optional feedback tensors are not
  themselves evidence of an inference crash. A nonzero process exit
  or explicit video decode failure still requires investigation.
