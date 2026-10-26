# Milestone 13A follow-up: diagnose OHP pose coverage before 1,921-clip extraction

## Confirmed user pilot — 8 publisher train + 8 publisher validation clips

```text
python -m scripts.extract_fitness_aqa_ohp --split train-val --max-videos-per-split 8
97 tests passed, 4 warnings.

Split   decoded    sampled    boundary    eligible    accepted  coverage  elbow+  knee+
train   8/8         65         6           59          13        22.0%     1      0
val     8/8         56         6           50          17        34.0%     1      0
```

No decode errors; the EGL ES3 fallback to ES2, TensorFlow Lite XNNPACK
and protobuf deprecation messages were not fatal.

**This is a small, biased pilot**, selected as the alphabetically earliest
8 videos from each publisher split. In particular, knee-error positive
accepted frames are **zero**, which does not mean no knee-error label
exists among all source OHP clips.

The OHP-111 feature function still requires original Squat **hips/knees/ankles**
visibility ≥0.5, then adds both wrists/elbows visibility ≥0.5. That may
exclude OHP frames with occluded legs even when shoulders/elbows are visible.
The distribution of reasons is currently unknown, and lowering *all*
visibility thresholds is not justified.

## New stratified diagnostic (no model train, no source TEST)

```bash
git fetch origin
git switch --track origin/feat/m13-ohp-stratified-pose-quality-audit
python -m pytest -q

# Target annotation-diverse publisher train/val clips:
python -m scripts.audit_fitness_aqa_ohp_quality \
  --videos-per-split 30 \
  --sample-fps 2 \
  --seed 42 \
  --min-visibility 0.5 \
  --alternative-visibility 0.35
```

Reads up to 30 **TRAIN** and 30 **VAL** clips from existing labeled
OHP ZIP, one at a time, with bounded temporary disk use. Selects an
initial approximately balanced quota of:
1. videos containing publisher `error_knees` intervals (may also have elbows);
2. videos with `error_elbows` intervals but none for knees;
3. videos with neither named annotation.

Selected clips are **not prevalence-representative**. Do not extrapolate
acceptance percentages to the full 2,260 assigned OHP clips.

Reports `artifacts/fitness_aqa_ohp_quality/diagnostic.json`:
- First failure reason (mutually exclusive): no pose, Squat-base-critical
  joint (shoulder/hip/knee/ankle) visibility, OHP-upper limb (elbow/wrist)
  visibility, invalid landmarks or degenerate geometry.
- Per-joint failure counts, which **may overlap** across frames.
- Eligible labeled-positive and labeled-negative frames for both tasks,
  split by actual `0.5` accepted/rejected status.
- Whether the **same MediaPipe output** could be accepted at a hypothetical
  `0.35` threshold, without retraining models or changing production data.
- Per-annotation-stratum counts, boundary exclusions, failed clips.

The implementation explicitly cross-checks its accept/reject reason against
the current `pose_to_ohp_features(..., min_visibility=0.5)` output on
each decoded eligible frame.

## Interpretation/decision gates

- **If ankles dominate** base-gate rejection, evaluate a separate
  nonuniform OHP policy with strict elbow, wrist and knee visibility.
  OHP `error_knees` still needs reliable knee observations; do not simply
  remove all lower-body landmarks or lower their threshold.
- **If wrists/elbows dominate**, investigate MediaPipe arm occlusions,
  frame/camera angle and per-joint scores before feature changes.
- **If no-pose dominates**, alternative pose extraction settings may help;
  a lower visibility threshold cannot recover frames with no pose.
- **If error_knees=1 frames are present among eligible but absent
  among accepted**, assess selection bias and joint failure reasons before
  launching model comparisons.
- A purely higher frame acceptance ratio is **NOT evidence** of better
  OHP error prediction; preserve original 0.5 gate and model versions
  until a controlled train-only calibrated val comparison shows benefits.

When quality issues are understood, run a **full source TRAIN+VAL
extraction** using the retained/frozen preprocessing, then build M13B
OHP-specific static and causal temporal RF benchmarks.
Do not decode, fit, or compare models on source TEST clips. Publisher
video-ID disjointness is not proven athlete-disjoint.
