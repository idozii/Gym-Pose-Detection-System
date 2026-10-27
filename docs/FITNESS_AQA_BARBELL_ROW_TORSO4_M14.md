# Milestone 14C - Barbell Row shoulder-hip-only static image pose extraction

## Evidence from the owner's M14B source-image audit

The dataset is IMAGE-based, with publisher train/val/test **specific to each
target**, not an OHP video sequence. The real original full extraction of
8-joint/32-feature images retained only 4.1-4.9% of task-selected images,
including just **8 positive lumbar validation** samples.

The owner ran PR #22's train/validation source-image annotation-stratified
audit (180 photos per task per split, 90 error-positive/90 error-negative):

| Task / split | All8>=.50 admitted | Four shoulder-hip joints>=.50 | Positive labels under four-joint gate |
| --- | ---: | ---: | ---: |
| Torso Angle / Train | 4/180 (2.2%) | **129/180 (71.7%)** | 61/90 |
| Torso Angle / Validation | 7/180 (3.9%) | **135/180 (75.0%)** | 64/90 |
| Lumbar / Train | 4/180 (2.2%) | **123/180 (68.3%)** | 63/90 |
| Lumbar / Validation | 9/180 (5.0%) | **129/180 (71.7%)** | 71/90 |

In the audited subset, low knee/ankle visibility accounts for large
losses with the old eight-joint gate, but between 44 and 55 of 180 images
per task/split also had **no pose detected**. A pose detector cannot be
fixed by reducing a visibility cutoff for undetected people.

**Crucial distinction:** these pilot percentages deliberately balance
labels, so they are *NOT estimates* of full publisher dataset coverage,
accuracy, AP or F1. Full train+validation extraction is the next evidence.

## What changes in this PR

We do **not** simply relax the old 32-feature model's eight-joint gate
while continuing to use the untrustworthy knees and ankles. Instead:

- New `barbell-row-torso4-image-20-v1` schema with the **same four
  shoulder/hip landmarks used in both the visibility gate and every
  feature calculation**. Twelve hip-centre/shoulder-distance-normalized
  `x,y,z` values + eight torso/width/relative-centre geometry proxies.
- Uses visibility>=0.50 (the successful audit candidate). No knee,
  ankle, wrist or elbow landmark is read by the new feature function.
- An opt-in `feature_profile="torso4"` reuses M14A's tested ZIP
  member selection, publisher *task-specific* train/val image splits,
  decoding and pose abstention rules. The default `full8` profile
  and all 32-feature CSV filenames remain unchanged.
- Wrapper CLI writes **separate Barbell Row torso4** CSVs/manifest
  to a new default folder; it cannot overwrite the old 32-feature
  outputs by default.
- One MediaPipe forward pass per source TRAIN/VAL image, even if an
  image is selected for both target tasks in the same source split.
- Publisher test image bytes are never read/decoded/used for threshold
  selection or model training. It only metadata-checks all publisher
  split IDs in the original ZIP.

### Run a 12-image-per-task pilot

```bash
git fetch origin
git switch feat/m14-barbell-row-torso4-extraction 2>/dev/null || \
  git switch --track origin/feat/m14-barbell-row-torso4-extraction
git pull --ff-only
python -m pytest -q

python -m scripts.extract_fitness_aqa_barbell_row_torso4 \
  --split train-val \
  --max-images-per-task-split 12 \
  --seed 42 \
  --min-visibility 0.5
```

The pilot samples source error-positive and source error-negative IDs in
each original publisher target split. It may have different retained
class prevalence after pose abstention. Confirm
`image_decode_failures: 0`, expected source schema and outputs.

### Full original TRAIN + VAL images (TEST remains locked)

```bash
python -m scripts.extract_fitness_aqa_barbell_row_torso4 \
  --split train-val \
  --max-images-per-task-split 0 \
  --min-visibility 0.5
```

Uses the already downloaded Barbell Row ~550 MB ZIP and source JSONs.
No new external data, downloads or OHP/Squat re-extraction required.
Full extraction replaces only new torso4 pilot CSVs, not original ZIP,
original 32-feature CSV, OHP/Squat data or pretrained models.

Output structure:

```text
data/features/fitness_aqa_barbell_row_torso4/
├── fitness_aqa_barbell_row_torso4_torso_angle_train_pose.csv
├── fitness_aqa_barbell_row_torso4_torso_angle_val_pose.csv
├── fitness_aqa_barbell_row_torso4_lumbar_train_pose.csv
├── fitness_aqa_barbell_row_torso4_lumbar_val_pose.csv
└── fitness_aqa_barbell_row_torso4_extraction_summary.json
```

## Integrity and reporting

For each source split and publisher target, report:
- Selected source image IDs (full TRAIN+VAL or balanced pilot).
- Successfully decoded images, pose rejected/accepted, decode failures.
- Original label-positive selected/accepted counts and negative accepted.
- Task-specific pose coverage = accepted / task-selected images.
- Candidate `image_id.rsplit('_', 1)[0]` prefix overlap across source
  train/val. This is a *heuristic*, NOT a verified source clip or athlete ID.

Both source labels remain per-image binary; never infer video order or
temporal sequences by sorting filename suffixes. A `0` means absence
of the named annotated error, not globally correct lifting technique.

### IMPORTANT: what the model cannot measure

The shoulder/hip-only pose can estimate coarse torso orientation and
camera-relative geometry. It **cannot measure curvature of the lumbar
spine**. If future torso4 models predict publisher `lumbar` source
labels, those are empirical label correlations, not direct anatomical
assessment, medical diagnosis or injury-prevention coaching.

No Static RF accuracy can be inferred until full train/val data, class
retention, near-duplicate/prefix provenance and a training-only
calibration protocol have been checked. DO NOT decode publisher TEST
or claim independently held-out athlete generalization at this step.
