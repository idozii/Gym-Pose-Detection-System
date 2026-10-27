# M28 - Push-up temporal robustness and feature-family ablation

## Observed M27 (original TRAIN only)

The owner completed M27 on **83 identical Push-up TRAIN clips from
14 original TRAIN participants**, with original M20 VAL and TEST
unchanged:

| Metric | Baseline 64 | Temporal 80 | Difference |
| --- | ---: | ---: | ---: |
| Balanced Accuracy (BA) | 54.18% | 58.91% | +4.73 percentage points |
| F1 on publisher bad label | 52.50% | 54.05% | +1.55 points |
| Recall on publisher bad label | **51.22%** | 48.78% | -2.44 points |
| Average Precision (AP) bad | 61.45% | 63.80% | +2.35 points |

**Conclusion:** This is a modest exploratory improvement in a small
TRAIN-only grouped CV. It is not proof of better or safer gym form
assessment, and recall on the bad label worsened.

The last independently evaluated protocol M26 showed worse performance
for the 64-feature upper-expanded model on original M20 VAL
(**51.89% BA on 23 paired clips**) than the full-admission baseline
(**61.36%**). On 7 upper-only VAL clips it missed **3/3 bad**.
These prior VAL observations were already used in research decisions.
**Do not reuse VAL to select new M27/M28 features or thresholds** and
do not open original TEST.

## M28 question

Is the 4.73-point M27 gain from interpretable elbow/wrist
kinematics, or is the model exploiting correlations between the
publisher good/bad label and video-framing/visibility metadata?

Four **predeclared** feature families from the existing local CSVs:

| Name | Feature count | What it includes |
| --- | ---: | --- |
| `baseline_64` | 64 | Original M23 pose median/std |
| `kinematic_76` | 76 | 64 + 12 elbow angle / angular speed / wrist motion |
| `quality_proxy_68` | 68 | 64 + 4 timeline duration, accepted fraction, accepted frame count, arm-side switching |
| `temporal_80` | 80 | All original 64 + 16 temporal features |

The 4 quality proxies are potentially correlated with camera
framing, clip length and pose visibility rather than biomechanics.
We do **not** assume these are harmful without results, but
their isolated evaluation helps interpret the mechanism.

Both CV strategies use the *identical* 83 clips, fixed 200-tree
Random Forest (min_samples_leaf=2, balanced classes,
random_state=42, cutoff=.5) and the frozen source TRAIN subjects:

1. **GroupKFold(5)**, matching M27 exactly.
2. **LeaveOneGroupOut**, holding out each original TRAIN subject
   in turn (14 folds).
3. For both strategies, **1000 paired participant-cluster
   bootstrap resamples** on the fixed out-of-fold predictions,
   comparing each alternative against `baseline_64`.
   These are **descriptive** bootstrap intervals, not external
   test confidence intervals. No model is refit inside resampling.

The default script checks that `baseline_64` and
`temporal_80` rerun the already observed M27 metrics
**exactly** before writing its new report, if the original
M27 JSON is found.

## Run locally without dataset ZIP or additional downloads

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

python -m scripts.m28_pushup_temporal_robustness
```

Inputs are existing **local** M20 manifest and accepted features,
M23 upper-body TRAIN features, and M27 temporal TRAIN features.
No video extraction, inference backend change, new hosting charges
or joblib artifact publishing is performed.

Only the following Git-ignored report is created:

```text
data/processed/m28_pushup_temporal_robustness.json
```

Print a compact, immediately comparable table:

```bash
python - <<'PY'
import json
p="data/processed/m28_pushup_temporal_robustness.json"
with open(p) as f: r=json.load(f)
print("Subjects:",r["train_people"],"Videos:",r["paired_train_clip_count"])
print("M27 parity:",r["m27_reproducibility"])
for cv,results in r["results"].items():
    print("\n",cv,"folds:",results["folds"])
    for name,item in results["models"].items():
        m=item["all_paired"]
        print(name,"BA:",round(m["balanced_accuracy"],4),
              "F1bad:",round(m["f1_bad"],4),
              "Recallbad:",round(m["recall_bad"],4),
              "APbad:",round(m["average_precision_bad"],4))
    print("Descriptive paired delta BA intervals:")
    for name,metrics in r["cluster_uncertainty"][cv]["paired_deltas"].items():
        print(name,metrics["balanced_accuracy"])
print("VAL/TEST accessed:",r["original_M20_VAL_or_TEST_used"])
PY
```

## Research decision

A temporal model should **not** be selected merely because of a
small 5-fold BA improvement. Check whether kinematic-only features
also improve LOSO BA/AP without worsening bad recall, and whether
the paired difference intervals are compatible with zero.

If nearly all apparent improvement comes from
`quality_proxy_68`, that suggests reliance on coverage/framing
cues. If LOSO reverses the 5-fold ranking, do not claim stable
generalization. With only 14 TRAIN people, even positive
paired intervals cannot replace genuine external validation.

For scientific progress beyond M28, acquire separately consented
videos from **new people not in Gym7**, create a frozen
evaluation protocol and only then perform a single external
comparison, measuring rejection rates and classifier performance.
The private original M20 TEST must remain uninspected.

No Push-up model in M28 is approved for the live single-frame
Fitness-AQA backend. The public site continues to provide
research visualization and its six original source-label models,
not validated technique correctness or injury-prevention advice.
