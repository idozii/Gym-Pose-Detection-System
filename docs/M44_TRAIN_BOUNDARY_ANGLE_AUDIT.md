# M44: TRAIN angle-observability and publisher-boundary audit

## Owner-run evidence motivating the experiment

M43 evaluated 20/20 original author TRAIN Squat videos, using the
unchanged source filename-hash seed 2026 and 312 source-labeled reps.
At dense 200ms (shared MediaPipe tracking history):

- Low-to-high original detector: TP=149, FP=11, FN=163.
- Of 163 unmatched GT repetitions, **132** had usable angle
  observations inside their publisher half-open interval but
  **no observed angle <=120 degrees inside that interval**.
- 22 had an observed low but no later high within that same interval;
  9 had a low-then-high sequence but were still unmatched.
- Only 2/312 publisher GT intervals contained <4 usable angle ticks.
- All 11 unmatched predictions lay **outside every GT event-end
  matching tolerance**.

At the every-other 400ms observation subsample:

- TP=118, FP=3, FN=194.
- 158 FN showed usable angles but no <=120 degree angle
  within their publisher GT interval.
- 117/312 source GT intervals had <4 usable angle ticks.
- All 3 FP predictions lay outside publisher event-end tolerances.

Paired ground truth rep outcomes: both matched=116;
dense-only matched=33; subsample-only matched=2;
neither matched=161.

These counts are TRAIN-only and observational; 149 true positive
events despite only 115 within-GT-window low-then-high observations
indicate that a detector event may match a publisher GT event end
without observing the whole low/high sequence inside its annotated
[start, end) interval. **Do not infer angle-threshold failure
from within-window FN counts alone.** The publisher interval
may omit relevant movement immediately before or after its bounds.
No source TRAIN Squat count=0 videos exist; M42 negative controls
were unavailable and specificity is unmeasured.

## Fixed M44 diagnostics (NO classifier changes)

M44 re-runs exactly the same source SHA256-selected TRAIN cohort
and frozen M41 and M43 JavaScript replay. Its extra measurements
are determined before looking at new M44 results:

1. For each unmatched publisher GT interval with at least one
   usable angle but **no angle <=120 degrees inside**, bin its
   *minimum observed within-window 2D knee angle* into:
   (120,135], (135,150], (150,165], or >165 degrees.
   The first boundary is the original M39 low threshold;
   the others are coarse **descriptive bins**, not parameter
   candidates, new decision thresholds or validated squat depth.
2. For exactly those same FN intervals, inspect **750ms before
   publisher GT start** and **750ms after publisher GT end**.
   750ms is the original M37 minimum GT matching tolerance.
   Report whether a <=120 degree observation is present
   before only, after only, both sides or neither side.
   Also count how many have *any usable angle* in each context.
   Absence of observations is not evidence no movement occurred.
3. For each unmatched detector prediction, inspect whether
   its timestamp falls *inside at least one publisher GT
   interval* or outside all GT intervals. This is distinct
   from the M43 comparison with **GT event-end tolerances**.

Every metric is reported separately for the original
200ms-target dense MediaPipe pose stream and every-other
400ms subseries. The source GT is original publisher CSV.
The matching algorithm and tie ordering remain unchanged.
The original M34 RepCycleCounter internal 400ms gate,
M39 geometric threshold, phase rules and tracking constraints
are never modified.

The M44 JS summary embeds M43/M41 matching checks. Python rejects
any result with inconsistent FN bin totals, missing fields, altered
source split, unexpected source counts, or mismatched FP totals.
Only **cohort aggregates** persist in the output JSON. Video
names, frames, raw landmarks and per-event times are not included.

## Critical caveats

- The two cadence conditions share a single 200ms MediaPipe
  tracking trajectory; subsampled poses are not independent
  400ms MediaPipe estimates.
- A 750ms context may overlap the *previous or next*
  publisher repetition. It is evidence of angle observations
  near boundaries, not proof that the same rep was recovered.
- Publisher [start,end) conventions, source video FPS/label
  synchronization, subject independence, camera angle and actual
  joint-angle calibration remain unverified.
- 2D camera-projected knee angles are not ground-truth joint
  kinematics or exercise technique and do not support injury-
  safety advice.
- Results are post-hoc exploratory TRAIN measurements, not
  independent validation, held-out TEST accuracy, significance,
  new model performance or a new baseline.
- Never adjust the detector against the three validation
  clips explored in M38 or borrow TEST media for analysis.

## Run with the existing Ubuntu project

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m44_train_boundary_angle_audit.py -q
npm test --prefix web

python -m scripts.m44_train_boundary_angle_audit \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --max-videos 20 \
  --root data/external/m44_train_boundary_audit \
  --report data/processed/m44_train_boundary_angle_audit.json
~~~

Reading the local ~8.9GB TAR.GZ involves publisher TRAIN
annotation and selected-video streaming passes; the full
archive is not extracted. At most 20 TRAIN Squat clips
are decoded. The output and clips are Git-ignored, and
existing M39-M43 reports are not overwritten.

## Aggregate M44 result inspection

~~~bash
python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path(
    "data/processed/m44_train_boundary_angle_audit.json"
).read_text())
print("Split:", r["source_split"])
print("Evaluated/excluded:", r["videos_evaluated"], r["videos_excluded"])
print("Reps:", r["annotated_reps"])
for condition, d in r["m44_boundary_diagnostics"].items():
    print("\n", condition)
    print("No low inside GT FN:", d["unmatched_gt_no_low_inside"])
    print("Minimum angle bins:", d["min_angle_bins_for_no_low_gt"])
    print("Low near start/end:", d["low_in_750ms_context_for_no_low_gt"])
    print("Usable before/after:",
          d["no_low_gt_with_usable_angle_before"],
          d["no_low_gt_with_usable_angle_after"])
    print("Unmatched event inside/outside GT:",
          d["unmatched_prediction_location"])
    m = r["frozen_m41_metrics"][condition]["methods"]["low_to_high"]
    print("Recomputed TP/FP/FN:", m["event_tp"], m["event_fp"], m["event_fn"])
PY
~~~

If the recomputed frozen M43 or M41 metrics differ from the
owner's existing reports, record that reproducibility
difference rather than silently changing the old numbers.
The purpose of M44 is to decide which *hypotheses need
controlled future TRAIN experiments* - not to optimize
thresholds based on this audit alone.
