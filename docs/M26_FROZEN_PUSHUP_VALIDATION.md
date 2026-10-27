# M26 - Predeclared original M20 Push-up validation comparison

## Decision from observed M25 LOSO

Owner M25 results on 14 original TRAIN Push-up participants and
69 paired clips:

| Model | LOSO BA | F1 bad | Recall bad |
| --- | ---: | ---: | ---: |
| Full paired | 54.96% | 50.79% | 47.06% |
| Upper paired | 56.39% | 51.61% | 47.06% |
| Upper expanded | 53.57% | 51.52% | 50.00% |

M24 prior TRAIN 5-fold results were 56.22%, 47.69%, 49.24%
balanced accuracy, respectively. **Relative ranking changed.**

M25 1000-resample subject-cluster bootstrap on fixed LOSO predictions:

- Upper paired minus Full paired delta BA interval: **-10.54 to
  +13.81 percentage points**, median +1.43 points.
- Upper expanded minus Full paired delta BA interval: **-8.73 to
  +5.97 points**, median -1.47 points.
- Both bad-label recall difference intervals also cross zero.

These are descriptive TRAIN-only data, not independent
confidence intervals. **No model has shown robust superiority.**
The upper gate was selected to improve pose extraction coverage,
not to claim improved bad-form recognition.

## Frozen M26 research question

**When models trained only on the ORIGINAL M20 TRAIN subjects are
compared on the SAME Push-up clips from the ORIGINAL M20 VAL
subjects, what is the quality/coverage tradeoff between the
original full-body admission and the expanded upper-body admission?**

Two predeclared candidates, to avoid choosing based on VAL:

| Model | TRAIN admission and schema | VAL scoring |
| --- | --- | --- |
| `full_paired` | M23 full-body admitted TRAIN clips, 64-feature schema | Only VAL videos both gates admit |
| `upper_expanded` | M23 all upper-body admitted TRAIN clips, same 64-feature schema | Exactly the SAME paired VAL video IDs |

Each uses 200-tree RF, min_samples_leaf 2, class_weight balanced,
random_state 42, n_jobs 1, prediction cutoff **0.5**.
No validation-driven threshold choice or hyperparameter tuning.

Metrics: paired VAL balanced accuracy, F1/precision/recall bad,
average precision bad, confusion matrices and by-view breakdowns.
New VAL clips admitted only by upper-body have a
**separate unpaired result**. Report gate coverage on all original
VAL Push-up manifest videos too. It is not valid to describe
upper-only results as paired accuracy improvement.

**Note about prior work:** M20 previously consulted these VAL
participants during earlier research. They are original
person-disjoint validation subjects, but **not a fresh, never-used
external holdout**. All later model choices after viewing VAL
are exploratory. Preserve the original TEST participants as
the FINAL untouched subject holdout.

## Run on Ubuntu with dataset already present

The original Mendeley Gym7 v3 Exercise-wise ZIP is needed for
the validation *extraction* step but will NOT be downloaded again.

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

# 1. Extract original VAL-subject Push-up video features once.
python -m scripts.m26_pushup_validation extract

# 2. Fit frozen RF candidates on TRAIN; score paired original VAL.
python -m scripts.m26_pushup_validation evaluate
```

Outputs are Git-ignored:

```text
data/features/m26_pushup_val_upper_v1/val_full_body.csv
data/features/m26_pushup_val_upper_v1/val_upper_body.csv
data/features/m26_pushup_val_upper_v1/val_extract_summary.json
data/processed/m26_pushup_frozen_validation_report.json
```

Both commands refuse to overwrite existing results. The
original M20 source video **TEST subjects are never decoded**.
No original M20 TRAIN video is re-decoded; the script uses the
existing frozen M23 TRAIN clip CSVs.

Use this snippet to summarize results:

```bash
python - <<'PY'
import json

p = "data/processed/m26_pushup_frozen_validation_report.json"
with open(p) as f:
    r = json.load(f)
print("Original VAL Push-up clips:", r["original_VAL_pushup_clips"])
print("Full-body admitted:", r["val_full_accepted"])
print("Upper-body admitted:", r["val_upper_accepted"])
print("Paired VAL clips:", r["paired_common_val_clips"])
for name, result in r["paired_model_scores"].items():
    x = result["all_common"]
    print(name, "BA:", x["balanced_accuracy"],
          "F1 bad:", x["f1_bad"],
          "Recall bad:", x["recall_bad"],
          "n:", x["n"])
print("Upper-only VAL:", r["unpaired_recovered_val_clips"]["all_views"])
print("Original TEST used:", r["original_TEST_scored"])
PY
```

No private video or model is committed. M26 never publishes a
model to Hugging Face/Render or enables real-time Push-up scoring.
The six existing Fitness-AQA named source-label models are unchanged.

## Before any future M27/production decision

The 64-feature M23 clip representation is a temporal aggregate,
while the production Vercel inference API takes **one frame** of
33 MediaPipe Tasks Vision landmarks. They are not compatible.
A correct realtime temporal model needs a buffer with explicit
window duration, frame visibility policy, backend latency,
browser/source landmark parity, consent and abstention tests.

If original VAL quality remains around chance, prioritize
new dataset participants, label verification and temporal features,
not deployment, aggressive RF tuning or TEST peeking.
