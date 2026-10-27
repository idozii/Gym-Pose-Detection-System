# M21 - Investigate front-view Push-up pose coverage before retraining

## Observed M20 extraction audit (owner data, 27 Oct 2026)

The owner ran the actual Gym7 M20 pipeline and compared the verified
video manifest against `m20_clip_features.csv`:

| Exercise / view | Available | Accepted | Rejected | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Bicep Curl - front, side, diagonal | 150 | 150 | 0 | 100% |
| Push-up - diagonal | 48 | 48 | 0 | 100% |
| Push-up - side | 48 | 48 | 0 | 100% |
| **Push-up - front** | **48** | **20** | **28** | **41.7%** |
| All | 294 | 266 | 28 | 90.5% |

Front Push-up rejection is equally distributed by the publisher good/bad
video labels: **14/24 bad and 14/24 good** rejected. This avoids a direct
global binary-label count imbalance caused by extraction, but **does not**
prove that missingness is random across participants, poses, viewpoint
variants or duration. Only 25 distinct subject IDs remain among accepted
clips; the number of available people in the rejected set may differ.

The existing M20 Push-up pose gate requires one **complete visible side**
of the body (shoulder, elbow, wrist, hip, knee and ankle) above .35
visibility, plus a nondegenerate torso scale and **at least four eligible
sampled frames per video**. With a front viewpoint, MediaPipe may not
be able to identify all these joints in enough frames. This is a
**hypothesis to measure**, not a proven reason.

Reported validation results are 60.0% balanced accuracy (Bicep Curl),
61.4% (Push-up), with low bad-label recalls (46.7% and 50%). These were
measured on **five held-out validation subjects**, a small sample, and
are not certified exercise correctness or real-world workout advice.

## Run a diagnostic on TRAIN subjects, with no re-extraction or retraining

On the owner's computer:

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

python -m scripts.m21_audit_front_pose
```

This script reads exactly these **already available local inputs**:

```text
data/external/mendeley_gym7/Dataset Exercise-wise.zip
data/external/mendeley_gym7/m20_manifest.csv
data/features/m20_gym7/m20_clip_features.csv
```

It reconstructs the original M20 seed-42 **subject-disjoint split** using
the accepted feature CSV and selects **only previously rejected front
Push-up videos from TRAIN participants** for diagnosis. Video IDs owned by
VAL or TEST participants are excluded before video decoding. It processes
one ZIP member at a time and keeps no frames or raw videos on disk after
each temporary extraction. All the JSON output contains is reason,
label and anonymous joint-count aggregates; no camera frames, source
media or individual pose coordinate records.

Outputs:

```text
data/processed/m21_front_pose_audit.json
```

To reduce diagnostics on a machine with limited time, pass
`--limit 6`, but the full TRAIN-only diagnostic is preferable.

Example JSON structure (**illustrative labels only, not observed
values**):

```json
{
  "by_label": {
    "bad": {
      "clips": 8,
      "sampled_frames": 900,
      "accepted_frames": 50,
      "reasons": {
        "no_pose": 200,
        "low_one_side_visibility": 600,
        "low_torso_scale": 50,
        "accepted": 50
      },
      "missing_joints": {
        "left_ankle": 300
      }
    }
  }
}
```

Actual counts are determined only by running the user's own data.

## How to decide M21 improvements (train only)

- High **no_pose** rate: check framing, clip orientation and whether the
  original MediaPipe Solutions Pose detector sees floor-level Push-ups.
  Explore different cropping / aspect treatment on TRAIN participants.
- High **low_one_side_visibility**, especially ankles and wrists:
  examine a *separate* feature/gate schema requiring visible key joints
  appropriate to the exercise/view. Do **not** simply make all missing
  joints up or loosen thresholds until validation looks good.
- High **low_torso_scale**: compare camera-relative torso normalization,
  avoid silently replacing the original schema with a different input
  for already trained models.
- High **accepted** but fewer than four eligible frames per video:
  examine sampling density and pose quality over time using new TRAIN-only
  extraction. Video-level and frame-level acceptance are different.
- If the failure is fixed, introduce a **versioned new M21 dataset
  feature schema**; extract the full dataset once under a NEW path,
  preserve original M20 files, keep original TRAIN/VAL/TEST subject
  assignment frozen and measure *coverage plus balanced accuracy*
  by view, by label and by participant.

**Do not inspect, tune on, or evaluate original held-out TEST subjects
until the final independent report.** Once the training gate is fixed
on TRAIN, validation can compare independently frozen candidates,
and only one final TEST evaluation is appropriate.

M21 only diagnoses coverage. It does **not** by itself add a Push-up
score to the public research website or turn publisher good/bad
video labels into anatomically specific form errors.
