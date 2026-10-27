# M23 - Frozen TRAIN-only Push-up schema pilot

## Evidence motivating M23

The owner's M22 paired front-view experiment on 15 previously rejected
original TRAIN Push-up videos sampled 513 frames. At identical baseline
MediaPipe complexity 1, the original complete-side full-body gate
accepted **24/513 frames and 0/15 videos**, versus **258/513 and
14/15 videos** when the knee/ankle visibility requirements were
removed. Heavy MediaPipe accepted 262 upper-body frames with the same
14/15 recovered videos; lower detection confidence accepted 270,
also 14/15 recovered. This strongly favors studying the *gate*
before increasing detector complexity.

**This is frame/clip coverage evidence, not a correctness-model accuracy
improvement.** No independent TEST performance has been measured.

## Two confounders that M23 explicitly controls

1. **Participant assignment drift**: M20's seed-42
   `split_by_subject(original_accepted_csv)` is reconstructed BEFORE
   extracting newly recovered samples. Re-running
   `split_by_subject` after adding accepted videos could change the
   subject membership of TRAIN, VAL and TEST, making metrics
   incomparable and potentially contaminating selection. M23 only
   decodes original TRAIN subjects from original M20; unseen or
   entirely rejected participants are quarantined.
2. **Feature schema drift**: M20's 264 features include 33 landmarks
   and their visibility, including inaccurate leg XYZ on low-visibility
   frames. A new **versioned 64-feature** schema represents only eight
   upper-body joints (L/R shoulder, elbow, wrist, hip), with 3 normalized
   coordinates and visibility per joint. Coordinates below the
   visibility threshold are zero-masked; upper-body gate chooses
   the visible side for torso anchoring. Each accepted clip
   concatenates the temporal median and standard deviation of
   the 32 frame features. This creates a **different model contract**:
   **do not load these features into any existing M20 RF.**

To compare the gate effect without confounding feature representation,
M23 extracts **full_body** and **upper_body** clip tables from the
**same original M20 complexity=1 MediaPipe inference stream**,
using the SAME new 64 features. Both require four accepted frames,
so a difference between their TRAIN coverage comes from the gate.

## Run now on the owner's Ubuntu machine

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

python -m scripts.m23_train_only_pushup_features
```

This reads only the same Mendeley ZIP, reviewed manifest and original
M20 accepted features you already have. It reprocesses **all Push-up
clips belonging to original TRAIN participants** (both previously
accepted and rejected) to avoid mixing feature-generation rules.

Expected local ignored outputs:

```text
data/features/m23_pushup_train_upper_v1/train_full_body.csv
data/features/m23_pushup_train_upper_v1/train_upper_body.csv
data/features/m23_pushup_train_upper_v1/train_extract_summary.json
```

Print a concise report:

```bash
python - <<'PY'
import json

p = 'data/features/m23_pushup_train_upper_v1/train_extract_summary.json'
with open(p) as f:
    r = json.load(f)
print('Schema:', r['schema'])
print('Selected original TRAIN clips:', r['selected_train_pushup_clips'])
for gate, record in r['gate_counts'].items():
    print(gate, record['accepted_clips'], record['by_label_view'])
print('File failures:', r['file_failures'])
print('Validation/TEST video decoded:', r['validation_or_test_video_decoded'])
PY
```

**The experimental code does not read or decode any original
validation/TEST videos.** It reads their IDs only to restore the
previously frozen split membership. It writes no images/videos and
does not overwrite the M20 original `m20_clip_features.csv` or
M20 model files.

The results remain on the user's local disk and are ignored by Git.
The tool refuses to overwrite an existing M23 output folder.
A pilot subset can use `--limit 20 --output-dir
data/features/m23_pushup_pilot20`, but the complete TRAIN
experiment is needed before drawing quantitative conclusions.

## What we do after the TRAIN result

1. Confirm TRAIN full-body accepted counts agree with M20 for
   the same original TRAIN videos, and calculate changes by original
   good/bad label and camera view.
2. Measure upper-body gate gains and pose rejection separately
   for front/side/diagonal and both classes; inspect missingness
   and accepted-frame duration bias.
3. Before performing ANY new VAL/TEST extraction, fix and document
   the selected feature schema, subject membership, frame sampling,
   Random Forest baseline, and comparison protocol on TRAIN.
4. Extract only the frozen VAL subjects for a **single predeclared
   validation comparison**, using the exact same schema.
   Compare balanced accuracy, recall, F1 and rejection rate using
   identical subject IDs and matched evaluation cohorts.
   No tuning using publisher TEST samples.
5. Once a scientifically supported clip-level Push-up quality model
   exists, add explicit real-time temporal-buffer inference and test
   Python MediaPipe Solutions versus browser MediaPipe Tasks feature
   parity, with an explicit ABSTAIN on bad framing. Do not add
   current new models to the live per-frame API.

M23 is a research-only **TRAIN extraction milestone**, not a
scientifically validated gym form coach.
