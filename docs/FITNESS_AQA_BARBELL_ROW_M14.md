# Milestone 14A — Fitness-AQA Barbell Row image pose extraction

## Why Row requires a separate pipeline

The original publisher Fitness-AQA Barbell Row labeled set is an **image
ZIP**, not the Squat/OHP video archive. Publisher labels are **per-image 0/1**:

| Target in code | Publisher source file | Labels |
| --- | --- | --- |
| `torso_angle` | `labels_torso_angle_error.json` | per image binary |
| `lumbar` | `labels_lumbar_error.json` | per image binary |

Unlike OHP and Squat, Barbell Row has **separate train/val/test image-ID
split lists FOR EACH TARGET**. Never naively create a unified split or
join two labels without checking split conflicts and original sample
provenance. The user's source M12 ZIP manifest previously passed with
**18,201 / 18,201** requested images present, no duplicate/missing
image stems and no cross-target exact-image split conflicts.

The image IDs resemble `56067_3_51`. A prefix like `56067_3` is
only a **candidate grouping heuristic**. It is not a verified source
clip ID or athlete ID. Cross-split near-duplicates and subject
dependence remain possible even when exact image IDs do not overlap.

## M14A code and research scope

- `src/gym_pose/barbell_row_features.py` adds dedicated, versioned
  `barbell-row-image-32-v1` static features: 8 *valid and visible*
  shoulder/hip/knee/ankle 3D MediaPipe landmarks (24 normalized x/y/z
  values) plus 8 camera-relative hip, knee, and trunk geometry proxies.
- `scripts/extract_fitness_aqa_barbell_row.py` reads only *requested*
  publisher TRAIN/VALIDATION image member bytes from the already
  downloaded `images.zip`. Uses `cv2.imdecode` and MediaPipe
  `static_image_mode=True` once per image, and writes separate task
  CSVs. Both tasks may share the same decoded TRAIN/VAL image, but the
  original split membership is preserved per task. TEST member bytes
  are never opened or decoded.
- Images failing decode, pose detection or the 0.50 gate on the eight
  required joints are not treated as negatives; task-local coverage
  and failures are reported.
- Zip central directory, original publisher labels and split ID lists
  are checked for integrity using M12. **No model is fit**, no
  prediction accuracy is claimed, no time-series ordering is assumed.

### Critical limitation of the lumbar label

MediaPipe's eight-joint shoulder/hip/knee/ankle representation does
**not directly measure lumbar spinal curvature**. A classifier trained
against the source `lumbar_error` label may learn **correlations**
with coarse body pose/camera viewpoint rather than true spinal
alignment. The benchmark must disclose this and avoid clinical
interpretation, injury prevention, or definite body-safety feedback.

## Prerequisites

- Your original M12 dataset files are already downloaded and verified.
- Optional camera dependencies are installed; if needed:
  `python -m pip install -e '.[camera,dev]'`.
- Archive path:
  `data/external/fitness_aqa/barbell_row/labeled/images.zip`.
- Task-specific split and label JSONs in:
  `data/external/fitness_aqa/barbell_row/labeled/annotations/`.

No additional download or extraction of 550-MB ZIP is required.

## Run pilot first

```bash
git fetch origin
git switch feat/m14-barbell-row-image-pose-extraction 2>/dev/null || \
  git switch --track origin/feat/m14-barbell-row-image-pose-extraction
git pull --ff-only
python -m pytest -q

python -m scripts.extract_fitness_aqa_barbell_row \
  --split train-val \
  --max-images-per-task-split 12 \
  --seed 42 \
  --min-visibility 0.5
```

Pilot samples up to 12 source images per task and split, balancing
named-error positive and negative annotations when available. It
does NOT represent true error prevalence or independent athlete count.

If the pilot exits with no image decoding failures, run full
official publisher TRAIN and VAL per task only:

```bash
python -m scripts.extract_fitness_aqa_barbell_row \
  --split train-val \
  --max-images-per-task-split 0 \
  --min-visibility 0.5
```

This overwrites only the earlier **new Barbell Row** pilot CSV files,
not the original ZIP/JSON, Squat/OHP data, or any trained models.
All data and derived outputs are ignored by Git.

## Produced local files

```text
data/features/fitness_aqa_barbell_row/
├── fitness_aqa_barbell_row_torso_angle_train_pose.csv
├── fitness_aqa_barbell_row_torso_angle_val_pose.csv
├── fitness_aqa_barbell_row_lumbar_train_pose.csv
├── fitness_aqa_barbell_row_lumbar_val_pose.csv
└── fitness_aqa_barbell_row_extraction_summary.json
```

Each CSV contains `image_id`, `split`, `target`,
`error_label`, unverified `candidate_clip_prefix`, and 32 named
numeric features, with distinct source publisher task splits.

The report shows requested/decoded images, decode failures,
accepted and rejected poses per target, retained class-positive
counts and pose coverage, and candidate filename-prefix overlap.
**The publisher TEST images are NOT decoded or scored.**

## Next: image-only Row static model, with leakage safeguards

After seeing the actual full extraction summary:

1. Confirm nonempty positive and negative accepted frames per task.
2. Inspect `candidate_clip_prefix` overlaps between publisher train
   and validation within each task. If they overlap, investigate
   source-clip identity before claiming video-disjoint generalization.
3. Build **separate** static RFs per publisher task and original split;
   do not make temporal features based only on ID suffix ordering.
4. Calibrate thresholds strictly on *training groups*. A filename
   prefix can be used as a **conservative candidate block** but is
   not scientifically verified video or athlete identity.
5. Report pose acceptance/abstentions and frame imbalance, plus
   sensitivity to image viewpoint. Full independent athlete
   generalization or performance on the untouched publisher TEST
   is outside M14A.

## OHP prior research result (kept separate)

The user's latest OHP five-seed train-calibrated study found
**Knee Temporal AP, Macro-F1 and Recall higher in 5/5 seeds**;
Elbow Temporal AP higher in 4/5 seeds, but elbow Macro-F1 and
precision unstable. These are correlations across seeds on the
**same repeatedly-used validation source**, not independent tests.
M14A deliberately does not change those models or thresholds.
