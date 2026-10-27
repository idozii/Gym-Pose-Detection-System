# M30 - Bicep Curl TRAIN-only temporal ablation + Push-up web preview

## What is shipped to the public website

Users can now select **Push-up - Local Preview** next to Squat,
OHP and Barbell Row. The local page:

- Uses the existing MediaPipe Tasks Vision pose detector and
  user's choice of front/rear camera and selfie preview mirroring.
- Shows 2D *camera-plane* elbow angle for the most-visible arm
  using shoulder, elbow, wrist and hip tracking.
- Shows the range of observed elbow angles in a local rolling
  **10-second window** once at least four usable samples exist.
  The window expires on inactivity and is reset when changing
  exercise, switching camera, stopping or hiding the tab.
- Explicitly says **NO QUALITY SCORE / NO REP COUNT / NO COACHING**,
  and refuses to show an angle when the elbow is missing, too
  uncertain or geometrically degenerate.
- Disables remote landmark-inference consent for Push-up.
  The network scoring client still accepts **only** Squat, OHP,
  and Barbell Row for six original Fitness-AQA source-label models.
  It is not modified to accept Push-up labels.

**Why no Push-up good/bad verdict?** M26 original validation
accepted 30/30 with upper-body pose gating but achieved
**51.89% BA** on 23 common clips; on seven additional
upper-only clips the model missed all three publisher bad labels.
M28 temporal feature improvements remain TRAIN-only and
all paired subject-bootstrap BA difference intervals include
zero. Video-quality RFs consume *whole-clip* 64/76/80-feature
summaries; the public backend accepts a single 33-landmark
snapshot and cannot serve them safely or meaningfully without
a separate temporal-window backend and cross-MediaPipe
feature parity. Consequently Push-up is visible now
**as a real-time geometry exploration, not an exercise-form
correctness predictor**.

No video frames are transmitted, stored or downloaded by
the website. MediaPipe WASM/model assets still load from
their existing public CDN/Google origin. The new motion
window runs locally and maintains only a short-lived set of
angle numbers in JS memory.

## Offline Bicep Curl research baseline

The publisher Gym7 archive already contains **150/150 valid
Bicep Curl clips** across front, side and diagonal, and
both publisher video-quality labels `good` and `bad`.
M20 owner VAL performance for Curl was roughly 60% BA,
below reliable automatic scoring.

M30 follows the M27 motion hypothesis **only on the original
seed-42 M20 TRAIN participants**, without opening, reading
or evaluating the M20 VAL/TEST videos. The original 264
M20 per-clip landmark median/std features remain frozen.
M30 extracts 16 motion summaries from the elbow/wrist
trajectory of only source TRAIN curl clips.

To detect reliance on framing rather than actual kinematics,
it compares four fixed Random Forest feature families:

| Model | Features | Role |
| --- | ---: | --- |
| `baseline_264` | 264 | Existing M20 features, new person-heldout reference |
| `kinematic_276` | 276 | 264 + 12 arm-angle/motion descriptors |
| `tracking_proxy_268` | 268 | 264 + 4 duration/pose-coverage descriptors |
| `temporal_280` | 280 | All 264 + 16 temporal descriptors |

All four use `n_estimators=300`, `min_samples_leaf=2`,
`class_weight="balanced"`, `random_state=42`,
`n_jobs=1`, threshold 0.5 and identical source TRAIN
video IDs. Evaluation includes both original TRAIN-only
5-fold GroupKFold and Leave-One-Subject-Out, with class
metrics broken down by camera angle. There is no TEST
use, no extra datasets, no Git-committed model or live
Bicep Curl quality classifier.

## Commands on your Ubuntu computer

Use your existing dataset ZIP and `.venv-m20`, without
downloading another archive:

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

# Only original Bicep Curl TRAIN videos are decoded.
python -m scripts.m30_bicep_curl_temporal extract

# Same original TRAIN clips, person-grouped CV.
python -m scripts.m30_bicep_curl_temporal compare
```

Expected **local Git-ignored** files:

```text
data/features/m30_bicep_curl_train_motion_v1.csv
data/processed/m30_curl_train_cv.json
```

Read concise study results:

```bash
python - <<'PY'
import json
p = "data/processed/m30_curl_train_cv.json"
with open(p) as f:
    r = json.load(f)
print("TRAIN participants:", r["subjects"])
print("TRAIN Bicep Curl clips:", r["paired_clips"])
for strategy, values in r["models"].items():
    print("\n", strategy, "folds:", values["folds"])
    for name, data in values["models"].items():
        metrics = data["all_paired"]
        print(name, "BA:", round(metrics["balanced_accuracy"], 4),
              "bad F1:", round(metrics["f1_bad"], 4),
              "bad recall:", round(metrics["recall_bad"], 4),
              "bad AP:", round(metrics["average_precision_bad"], 4))
print("Original VAL/TEST scored:",
      r["original_M20_VAL_or_TEST_scored"])
PY
```

The extractor **fails without writing a partial dataset**
if any original TRAIN curl clip has insufficient time-order
feature samples. Do not "fix" failures by examining VAL/TEST
or changing the cohort silently. Instead inspect TRAIN-only
framing/geometry and version any new feature rules.

**No M30 model accuracy on the user's actual videos has
been observed in this PR.** CI only checks synthetic cases.
A model would require proper independent validation
before being described as an effective gym technique
or injury-prevention evaluator.
