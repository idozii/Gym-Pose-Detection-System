# M24 - Fair Push-up grouped cross-validation inside original TRAIN

## M23 real owner result

M23 succeeded on **84 original TRAIN Push-up clips**:

| Gate | Original TRAIN clips accepted | Acceptance |
| --- | ---: | ---: |
| Full-body | 69/84 | 82.1% |
| Upper-body | 83/84 | 98.8% |

Breakdown by original publisher label and camera view:

| Label / View | Full-body | Upper-body |
| --- | ---: | ---: |
| bad / diagonal | 14 | 14 |
| bad / front | 6 | 13 |
| bad / side | 14 | 14 |
| good / diagonal | 14 | 14 |
| good / front | 7 | 14 |
| good / side | 14 | 14 |

**14 previously rejected clips were recovered** by the upper-body
admission policy: seven `bad/front` and seven `good/front`.
There was no ZIP decoding error. This is clip acceptance,
NOT classification correctness. All files remain on the owner's
private local disk and **none of the M20 VAL/TEST video footage
has been decoded by M23**.

M24 asks: *Does the new upper-body clip representation retain or
improve recognition of the publisher's good/bad video labels on
subject-heldout TRAIN cross-validation, or does it just admit more clips?*

## Protocol

M24 reads the M23 64-feature `train_full_body.csv` and
`train_upper_body.csv` together with the **original** M20 accepted
`m20_clip_features.csv` and verified original `m20_manifest.csv`.
It rebuilds the **original seed=42 TRAIN subject membership** and
checks the identity and labels of every M23 row against the
original participant-reviewed manifest.

Within original TRAIN participants only, it runs **5-fold
GroupKFold by participant**, yielding out-of-fold predictions for
each baseline model. It compares:

1. `full_paired`: train 200-tree RF on the full-body admitted
   clips of the training participants; evaluate only held-out
   common full-body clips.
2. `upper_paired`: train 200-tree RF on **those same paired video
   IDs**, extracted under the upper-body gate; evaluate on the
   **same** held-out common clips.
3. `upper_expanded`: train 200-tree RF on **all eligible upper-body
   clips belonging to the training participants**, including
   recovered clips; evaluate on the SAME common held-out video IDs
   as models 1 and 2.

The original same-subject division is applied to every mode.
The 69 paired videos provide a **fair denominator** for direct
accuracy comparisons. Extra recovered held-out video predictions
are reported **separately** for upper-expanded only; they cannot be
compared directly with the full-body predictor, which abstains
on that subset. All metrics use publisher **video-level good/bad**
labels; no named elbow, knee or anatomical error can be inferred.

**Important limitations:** a five-fold TRAIN-only CV on roughly
14 subjects is a small pilot, not an independent validation
and definitely not a final TEST result. Gate changes also alter
which temporal frames are aggregated for the same video; the
paired video IDs control source content and subject membership,
not every possible extraction confounder.

## Run locally on Ubuntu

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate
python -m scripts.m24_pushup_group_cv
```

The script uses only small M23/M20 CSV files, **does not need the
2.9 GB source ZIP**, and does not decode camera media. All output
is aggregated metrics, no per-person predictions or videos.
It refuses to overwrite the original report if run again.

Output is Git-ignored:

```text
data/processed/m24_pushup_group_cv.json
```

Summarize report for a useful before/after table:

```bash
python - <<'PY'
import json
path = 'data/processed/m24_pushup_group_cv.json'
with open(path) as f:
    r = json.load(f)
print('M24 TRAIN subjects:', r['frozen_original_train_subjects'])
print('Original TRAIN clips:', r['original_train_pushup_clips'])
print('Paired videos:', r['paired_common_clips'])
print('Upper-only recovered videos:', r['newly_accepted_train_clips'])
for name, comparison in r['metrics_on_identical_paired_clips'].items():
    s = comparison['pooled_group_oof']
    print(name, 'balanced_accuracy:', s['balanced_accuracy'],
          'F1_bad:', s['f1_bad'], 'recall_bad:', s['recall_bad'],
          'n:', s['n'])
print('Upper-only out-of-fold:',
      r['recovered_unpaired_oof_only']['all_views'])
print('Original VAL/TEST scored:', r['heldout_m20_val_or_test_scored'])
PY
```

**Do not run `--evaluate-test` or rewrite original M20 artifacts.**
After TRAIN-only cross-validation, the next experiment should fix
candidate hyperparameters and the selected feature/gate schema
before extracting original VAL participants under the same
versioned pipeline. Then compare on a predefined matched
validation population with pose rejection reported alongside
model accuracy. The final independent TEST must remain
unseen until a documented final protocol is frozen.

The site continues to support only six original Fitness-AQA
source-error classifiers. M24 does **not** add a Push-up
real-time classifier or safety advice.
