# M27 - Push-up motion features from frozen TRAIN only

## What M26 established

Owner ran M26 on **30 original Push-up VAL videos**:

| Predeclared candidate | Clips covered | BA on 23 paired VAL clips | F1 bad | Recall bad |
| --- | ---: | ---: | ---: | ---: |
| Full paired, 64 features | 23/30 | **61.36%** | **57.14%** | 50.00% |
| Upper expanded, 64 features | 30/30 | 51.89% | 56.00% | **58.33%** |

Seven VAL clips were **upper-only recovered**, with original good/bad
counts 4/3. The upper model correctly predicted 3 of 4 good but
**0 of 3 bad**. Its recovery-set confusion matrix was
`[[3, 1], [3, 0]]`, so recall bad=0%, BA=37.5%.

On the **combined 30 VAL videos**, including seven recovered clips,
the upper-expanded model's calculated confusion matrix is
`[[8, 7], [8, 7]]` (15 good, 15 bad), hence BA=50.0% and
F1 bad approximately 48.3%.

These results rule out presenting the current upper-expanded model as
a reliable form-correctness classifier, even though it has much
better pose visibility coverage. Original M20 TEST has **not**
been accessed.

The original M20 VAL subjects were already used in earlier
exploration, so M26 is not pristine external validation. Viewing
these M26 results creates additional model-selection risk.
**No subsequent experiment can be presented as prospectively
validated by M26 without acknowledging that adaptive history.**

## Hypothesis (exploratory)

Existing M23 clip features contain **median and standard deviation**
of pose coordinates. They do not explicitly encode whether the
elbow flexes and extends in the correct time order, the amplitude
of the elbow angle, or how quickly the movement changes direction.

M27 tests a motion-aware augmentation using exactly the previously
accepted **83 original TRAIN Push-up videos**:

- Original 64 M23 clip features from upper-body admission.
- 16 time-series features from front/side/diagonal camera videos:
  elbow angle percentiles, standard deviation and range; elbow
  angular speed and direction changes; temporal first/last
  angle difference; normalized wrist-shoulder distance dynamics;
  valid timeline duration; accepted fraction, frame count,
  and selected left/right arm switches.
- The best visible upper side (shoulder, elbow, wrist, hip)
  is selected per accepted frame. At least four valid frames
  are required per clip. Gaps in the sampled timeline are
  preserved when calculating angular speed.
- No video, frame, raw landmarks, person IDs or per-video
  predictions are exported to the Git-tracked repository.

The procedure is not yet a scientific model improvement: temporal
descriptors from 2 FPS are a **hypothesis** and may alias fast reps.
They can also learn camera-specific visibility artifacts rather
than exercise quality. Controlled subject-separated comparison
is essential.

## Run M27 on the owner's Ubuntu machine

Use the **same original** Gym7 v3 ZIP, verified M20 manifest, and
M23 extracted upper-body feature CSV. No new download needed.

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

# Extract 16 additional motion features from frozen TRAIN subjects.
python -m scripts.m27_pushup_temporal_train extract

# Compare original 64 vs 64+16 on paired TRAIN videos.
python -m scripts.m27_pushup_temporal_train compare
```

Output files remain Git-ignored:

```text
data/features/m27_pushup_train_motion_v1.csv
data/processed/m27_pushup_temporal_train_cv.json
```

The extractor **fails without writing a partial CSV** if it cannot
compute temporally valid features for all 83 M23 upper-body
accepted TRAIN video IDs. This avoids changing the evaluation
cohort to artificially improve accuracy. If this happens,
send the exact error; do not lower thresholds by inspecting
VAL/TEST video.

The comparison uses fixed `GroupKFold(5)` with the
same original TRAIN participants, identical video IDs and
200-tree Random Forest, `min_samples_leaf=2`,
`class_weight=balanced`, `random_state=42`,
`n_jobs=1`. Every original participant is held
out from training for their respective fold. **Neither
VAL nor TEST videos are decoded or scored.**

Print a concise report:

```bash
python - <<'PY'
import json

with open("data/processed/m27_pushup_temporal_train_cv.json") as f:
    r = json.load(f)

print("TRAIN participants:", r["subjects"])
print("Same video cohort:", r["paired_clips"])
for name, result in r["metrics"].items():
    m = result["all_paired"]
    print(name, "BA:", round(m["balanced_accuracy"], 4),
          "F1 bad:", round(m["f1_bad"], 4),
          "Recall bad:", round(m["recall_bad"], 4),
          "AP bad:", round(m["average_precision_bad"], 4))
print("Original VAL or TEST used:", r["original_M20_VAL_or_TEST_used"])
PY
```

## Stop/go scientific requirements

1. Compare on identical original TRAIN clips and participant groups.
   An improvement in this group-only CV may be useful for research,
   but does **not** undo the poor M26 validation result.
2. Examine per-view class balance and rejection bias; do not
   select features or tune thresholds using M20 TEST.
3. A future independent study needs **new consented participants**
   or a genuinely untouched dataset, with preregistered
   feature schema, thresholds and model decisions.
4. Push-up webpage scoring requires a real-time temporal
   buffer and parity between browser MediaPipe Tasks Vision
   and training-side Solutions Pose, plus an ABSTAIN policy
   for low-quality frames. None are implemented in M27.
5. Fitness-AQA's six original error classifiers remain
   separate from these video-level Gym7 `good/bad` tasks.

M27 is exploratory **offline research**, not injury-risk
diagnosis, physical therapy guidance or a validated workout coach.
