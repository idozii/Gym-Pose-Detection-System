# Milestone 16B - explain live pose abstentions & measure runtime

## Owner-run M16A evidence

On Ubuntu desktop (Intel Mesa EGL/MediaPipe XNNPACK), the owner ran
`scripts.run_fitness_aqa_research_viewer` in Squat mode for 60 frames:

- **180 passed, 4 existing scikit-learn test warnings.**
- **6/6 original source-label RF model bundles** loaded successfully.
- Camera and MediaPipe successfully initialized; EGL, feedback-manager,
  protobuf deprecation and Wayland warnings were non-fatal.
- `frames_processed=60`; two selected Squat source-error detectors
  per frame produced **task_scores_emitted=0**, **task_abstentions=120**.
- Zero frames saved, zero alerts, publisher TEST untouched.
- Thus **120 = 60 frames × 2 source-error detectors**, not 120 camera frames.

This result **does not establish** whether MediaPipe detected no pose
or the original Squat source feature gate rejected detected pose.
The strict Squat `squat-103-v1` source vector requires **all 33
landmark coordinates finite** plus visibility >=0.50 of shoulders,
hips, both knees and ankles. No model retraining or threshold relaxing
is justified before knowing which condition failed.

## What M16B changes

The research viewer gains detailed **aggregate only** live metrics:

- Number of frames with/without MediaPipe pose landmarks and
  pose detection rate.
- Per-exercise frame counts, per-original-source-task scored counts
  and abstentions **by reason**.
- Non-identifying visibility failure counts for required anatomical
  joints (e.g., left_ankle, right_knee), not raw landmark coordinates.
- Distinction among:
  - `no_pose_detected`: pose landmarks absent entirely;
  - `required_joint_low_visibility`: at least one pose gate joint below
    0.50; joint counts included;
  - `invalid_landmark_count`, `invalid_landmark_values`,
    `nonfinite_landmark_values`, `degenerate_shoulder_width`;
  - `feature_gate_rejected_after_basic_checks`: extra features or
    other checks rejected a pose despite basic landmark quality;
  - `model_not_loaded_or_unavailable` or
    `model_inference_error`: loaded-model failures remain distinct.
- Aggregate mean/p95/max for pose processing, model scoring and
  overall frame time, plus **observed FPS including capture and
  OpenCV display**. Latency distributions cover the most recent
  3600 frames (bounded memory), not the entire session for long runs.
- Small live text hint on whether pose is missing or particular
  required joints are low-visibility.

The registry's feature schemas, thresholds and actual quality gates
are **unchanged**. These supplemental diagnostics NEVER admit an
otherwise rejected frame. They do **not** save images, source
annotations, per-frame predictions, user coordinates, personal data,
or recorded video. The printed summary is only counters and timings.

## Owner commands

```bash
git fetch origin
git switch feat/m16b-live-pose-abstention-profiling 2>/dev/null || \
  git switch --track origin/feat/m16b-live-pose-abstention-profiling
git pull --ff-only
python -m pytest -q

python -m scripts.run_fitness_aqa_research_viewer \
  --exercise squat --camera 0 --max-frames 120 \
  --trust-local-artifacts
```

For a camera framing check, **show shoulders, hips, both knees and
ankles**, keep the entire subject clearly inside the picture and
ensure sufficient light. The exact view is a qualitative debugging
step, NOT a source-validated camera requirement.

If you wish to compare availability across tasks, use keys
`1` Squat, `2` OHP, `3` Barbell Row. Note that selecting an
exercise changes which 2 source detectors are scored; it is not
automatic exercise recognition.

## Interpreting the result

1. `no_pose_frames` close to `frames_processed`: the issue is upstream
   of RF; investigate camera direction, framing and lighting, and the
   availability of MediaPipe pose landmarks. No threshold retuning.
2. High `required_joint_low_visibility` with knees/ankles:
   the source Squat gate is abstaining appropriately; adjust framing
   and view of the full body. Do not fabricate data or silently lower
   visibility while keeping untrustworthy joints in the feature vector.
3. `feature_gate_rejected_after_basic_checks`: inspect feature input
   validity and geometry separately, with no real-image capture.
4. `model_inference_error` with quality gates passing: verify registry
   artifact schema, installed sklearn version and predict-proba
   execution. Report the aggregate counters, not webcam imagery.
5. `observed_fps_including_display` is **not** a standalone benchmark
   of model latency. Compare the component timing distributions
   and account for frame capture and GUI costs.

All original publisher TEST data remains locked; the original source
validation data was repeatedly consulted during research and is not a
new independent benchmark. Squat Knees Forward source-validation
FPR **37.53%** and Knees Inward recall **25%** are insufficient for
dependable fitness-coaching alerts. Viewer still intentionally has
NO alarms, rep counting, or recommended technique corrections.

M16B is a runtime diagnostic, not an endorsement of medical
or injury-prevention advice, or clinical lumbar-spine inference.
