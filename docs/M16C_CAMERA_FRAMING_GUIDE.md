# Milestone 16C - live camera framing checklist for source-pose quality

## Grounded in the owner's real M16B 120-frame live webcam result

The M16B diagnostic script passed **186 tests, 4 existing sklearn test
warnings**, loaded **all six** trusted local source-error RF models, and
processed **120/120** local camera frames with MediaPipe **pose present
in every frame**. It produced **0 / 240** Squat source-label scores and
240 abstentions because required joints were below visibility >=0.50.

Among the original 120 frames with pose, low-visibility joint frame
counts for either Squat model were:

| Joint | Source-frame low-visibility count |
| --- | ---: |
| Left ankle | **120 / 120** |
| Right ankle | **120 / 120** |
| Right knee | **120 / 120** |
| Left knee | 98 / 120 |
| Left hip | 40 / 120 |
| Right hip | 40 / 120 |

This does **NOT prove** which objects were visible in the owner's
room or exact camera positioning: no webcam photos were retained
or supplied. However, it strongly suggests the selected full-body
Squat source pose gate is incompatible with the framing (e.g. a
desk-level camera only seeing upper body, cropped legs, occlusion or
low-confidence legs). MediaPipe **human detection was 100%**, so
retraining the RF or relaxing source visibility now is unwarranted.

Observed rate **15.49 FPS** including webcam capture and display;
mean MediaPipe **27.552ms**; mean model stage **0.317ms**, but that
stage returned without RF inference because every frame abstained.
True per-model execution latency has NOT yet been measured on
pose-admitted live frames.

## New on-screen framing checklist

The M16C `src/gym_pose/framing_guide.py`:

- Shows a dedicated **296-pixel-wide, separate right sidebar** with
  source-required joint names and transient MediaPipe visibility
  values (`VISIBLE` at >=0.50, `LOW` otherwise).
- Does NOT crop, stretch, pad, overwrite or modify **camera input to
  MediaPipe or model**. The expanded canvas is solely for the local
  display AFTER scoring; a display-only image is never saved.
- Squat: requires both shoulders, hips, knees and ankles.
- OHP: displays the union of elbow-task upper8 and knee-task lower8.
  These targets continue to have **separate, exact** quality gates.
- Barbell Row: displays shoulders+hips only; it does not
  pretend knee/ankle visibility is required for the torso4 model.
- If pose is missing or a visibility score is invalid, shows
  `NO POSE` or `INVALID`, rather than misreporting a valid pose.
- Original 6 source-label RF models, trained thresholds, feature
  schemas, `abstain` logic, privacy and research-only disclaimer
  are all unchanged.

**The guide is for webcam framing only**, not a traffic-light
movement-quality monitor. It does not measure safe biomechanics
or exercise correctness.

## Run

```bash
git fetch origin
git switch feat/m16c-live-camera-framing-guide 2>/dev/null || \
  git switch --track origin/feat/m16c-live-camera-framing-guide
git pull --ff-only
python -m pytest -q

python -m scripts.run_fitness_aqa_research_viewer \
  --exercise squat --camera 0 --max-frames 180 \
  --trust-local-artifacts
```

First adjust the camera view to clearly show both shoulders, hips,
knees and ankles. More distance and a lower/wider camera angle may
help if the original view only showed upper body. Stand so both feet
stay fully in view throughout Squat motion; maintain sufficient
light and avoid furniture obstructing the lower body.

The side panel displays which source-required joints are `LOW` and
`VISIBLE`. Continue until all the important Squat joints have
visibility >=0.50 at once. This DOES NOT guarantee correct technique
or independent model validity; it merely makes it possible to pass
the original source-model admission gate.

For an original-sized viewer without the sidebar:

```bash
python -m scripts.run_fitness_aqa_research_viewer \
  --exercise squat --camera 0 --max-frames 120 \
  --no-framing-guide --trust-local-artifacts
```

## What to compare after the local run

The JSON output retains the previous M16B `diagnostics` fields:
`pose_detected_frames`, `scored_by_target`,
`abstentions_by_target_and_reason`,
`required_low_visibility_joint_occurrences`,
`observed_fps_including_display`, and timing distributions.

A meaningful *framing progress* check is whether the count of
`required_joint_low_visibility` and each ankle/knee failure declines,
and whether any `task_scores_emitted` becomes nonzero **without
relaxing the original model gate**. A camera scoring success is NOT
evidence the experimental source-error model is accurate on this
different person/camera.

The original publisher TEST has remained untouched throughout
M16A/B/C. No local camera frames or per-frame source error estimates
are persisted. The original validation scores have been consulted
many times, so those numbers remain exploratory.
