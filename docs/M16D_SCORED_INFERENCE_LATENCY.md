# Milestone 16D — true scored-only inference latency and per-task acceptance

## Owner-observed M16C evidence

The owner ran the integrated webcam viewer with `--exercise squat
--max-frames 180` and the on-screen frame guide. Local results:
- **192 passed**, 4 pre-existing sklearn warnings; six trusted local
  publisher-source error RF bundles loaded.
- **180/180 frames had MediaPipe pose landmarks**.
- **47/180 frames** passed the *original* Squat source feature gate
  (26.1%); **94 model source-label scores** from two Squat targets.
- **133/180 frames** were abstained on by both Squat targets
  (266 source-label abstentions).
- Low visibility: right ankle **131/180**, left ankle **21/180**, right
  knee **14/180**. The right ankle remains the main pose bottleneck,
  but its cause cannot be determined from aggregate diagnostics alone.
- Overall viewer **15.8 FPS**, pose stage mean **25.725ms**, combined
  source-task model stage mean **6.344ms**, total displayed frame mean
  **62.886ms**.

The 6.344ms 'model_scoring' stage is averaged over BOTH admitted and
abstained frames. It does not separately establish real RF model-call
latency. Nor do any scores prove live form-assessment accuracy or
independent subject generalization.

## Implementation

- `gym_pose.research_viewer.analyze_landmarks` now records a
  monotonic elapsed interval around **each** task call to
  `ModelRegistry.score_landmarks`. This includes exact source feature
  construction, source quality gate and, if admitted, the classifier's
  `predict_proba` call. Each detector has independent timings.
- `LiveSessionDiagnostics` keeps a bounded rolling window of
  elapsed milliseconds for **successful experimental source-label
  score** calls and **abstained** calls, globally and by
  `exercise/target`. Max 3,600 recent calls for each window.
  It prints mean/p95/max instead of raw samples.
- Adds `source_label_score_admission_by_target`, including
  `source_frames_seen`, `scores_emitted`, `abstentions`, and
  `score_admission_rate`. The latter is a **pose+inference acceptance
  fraction, not classification accuracy**. It is valid for describing
  webcam availability only, on the selected source task.
- Mixed OHP visibility (elbow target scores, knee target abstains)
  is handled per target instead of attributing one frame's combined
  model time to both successful classifiers.
- Still local webcam-only, no per-frame predictions/poses/photos,
  no publisher TEST use, no model retraining, no thresholds changed,
  no technique alerts or rep counting. No extra webcam permission
  or dataset download.

The viewer still reports its original aggregate
`timing_ms.model_scoring` for backwards compatibility; the new
`timing_ms.successful_model_task_inference` and
`timing_ms.abstained_model_task_call` must **not be conflated**.

## Run from existing locally trusted RF files

```bash
git fetch origin
git switch feat/m16d-scored-inference-latency 2>/dev/null || \
  git switch --track origin/feat/m16d-scored-inference-latency
git pull --ff-only
python -m pytest -q

# Keep both ankles in view to get actual admitted RF calls.
python -m scripts.run_fitness_aqa_research_viewer \
  --exercise squat --camera 0 --max-frames 180 \
  --trust-local-artifacts
```

When complete, inspect JSON `diagnostics`:

```text
source_label_score_admission_by_target
timed_successful_task_calls_in_window
timed_abstained_task_calls_in_window
timing_ms.successful_model_task_inference
timing_ms.abstained_model_task_call
timing_ms.by_target_successful_task_inference
timing_ms.model_scoring
observed_fps_including_display
```

A model admission ratio higher than 26% for Squat would suggest
improved framing continuity, NOT model quality improvement. RF latency
cannot be judged from zero successful task calls; new empty timing
fields will correctly contain `null` rather than a fake 0ms.

On changing exercises with 1/2/3, aggregate per-target rates use
**the number of frames during which that exercise was selected**
rather than all frames in the session. No need to change existing
source quality thresholds to maximize availability.

## Beyond M16

This is a research-software final runtime audit. After the owner
confirms a green test suite and meaningful per-task scored latency,
the main remaining work is M17 documentation, reproducibility,
historical PR reconciliation, pipeline smoke tests, privacy,
versioning, and a research-limits demo/release.

Do not market these six source-label classifiers as reliable
autonomous form correction: Squat knees-forward source-val FPR ~37.5%,
Squat knees-inward source-val recall ~25%, Row lumbar source-val
recall ~36.7%, and real-camera domain shift is unmeasured.
Publisher TEST remains locked until a frozen legitimate protocol is
agreed. The repeated source validation analyses were exploratory.
