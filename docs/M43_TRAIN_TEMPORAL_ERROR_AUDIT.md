# M43: Frozen TRAIN-only temporal error audit

## Why M43 exists

The owner completed the M41 paired-cadence experiment on 20
SHA256-selected publisher TRAIN Squat videos, with 312 source
repetitions and 0 excluded clips:

| M39 low-to-high | 200ms observations | 400ms every-other subsample |
| --- | ---: | ---: |
| TP / FP / FN | 149 / 11 / 163 | 118 / 3 / 194 |
| Event precision | 93.13% | 97.52% |
| Event recall | 47.76% | 37.82% |
| Event F1 | 63.14% | 54.50% |
| Mean absolute count error/video | 7.90 | 9.55 |

On the shared dense MediaPipe tracking history, denser
observations decreased per-video count error for 10 videos,
tied for 9 and increased for 1. These data are TRAIN-only
and do not establish sampling causality or generalization.

M42 then established **no publisher-labelled zero-rep
Squat examples in LLSP/annotation/train.csv**. The M42
negative-control experiment appropriately stopped; no
negative-video false-alarm measurement exists. Do not
borrow validation or TEST clips or infer 0% false-alarm
rate from the absence of controls.

The M43 research question is: **Where do the original,
frozen low-to-high detector's remaining 163 TRAIN GT misses
and 11 unmatched predictions fall in the observed temporal
pose and annotation record?** The goal is descriptive
failure analysis before any new detector design.

## Strictly unchanged methodology

- Publisher source: owner's local authorized RepCountA.tar.gz.
- Dataset: publisher TRAIN annotation CSV and corresponding
  TRAIN video members only; validation/TEST untouched.
- Cohort: same filename SHA256 policy and seed 2026 as M40/M41.
  Limit 20, no new/random or outcome-chosen selection.
- Pose: Python MediaPipe Solutions 200ms-target trajectory,
  with every-other sample used as the 400ms observation control.
- Frozen methods: M34 with original 400ms internal gate,
  M39 low-to-high with original thresholds, and always-zero.
- Event policy: exactly original M37 nearest GT-end,
  one-to-one greedy matching, including tolerance caps.
- Events and poses are processed only in transient local
  files; only cohort-summed error counts are retained.

## M43 classifications

For **unmatched GT intervals** (false negatives), place each
publisher half-open [start, end) annotation window in exactly
one descriptive bucket:

1. No usable knee angle inside the publisher GT window.
2. Some usable angle, but none at or below frozen low angle.
3. At least one low angle, but no subsequent high-angle
   observation inside the same GT window.
4. Low then high observed inside the window, but no
   detector prediction was matched to that annotated rep.

Do not interpret these mutually exclusive buckets as
causes: an initial or return high pose can fall outside
the publisher interval. The low/high observation sequence
does not by itself prove that timing, amplitude, limb
consistency or the 400ms confirmation condition was met.

For **unmatched predicted events** (false positives), classify
whether their timestamp lies within the fixed end-matching
tolerance of *any* publisher GT event or outside all GT
end-matching tolerances. An unmatched prediction within a
tolerance may coexist with another closer prediction matched
to that same GT event. These are not technique labels.

M43 also aggregates GT-window coverage <4 and <6 usable
angles, any low, low-then-high, pending side switches and
tracking gap resets, and four **paired annotated-event
outcomes** between the shared-tracker 200ms and 400ms
conditions: both matched, dense only, sparse only, neither.
These paired counts sum to annotated GT repetitions.

All categories must reconcile exactly with the **same**
M41 TP/FP/FN totals recomputed on this rerun. M43 includes
new regression tests that enforce this contract. The
cohort-level M41 metrics are also recorded for comparison
with the owner-produced M41 report. If reproducibility
differs due to pose-estimator nondeterminism, report it
rather than rewriting previous metrics.

## Run locally on Ubuntu

From the project root:

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m43_repcount_train_error_audit.py -q
npm test --prefix web

python -m scripts.m43_repcount_train_error_audit \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --max-videos 20 \
  --root data/external/m43_train_temporal_errors \
  --report data/processed/m43_train_temporal_error_audit.json
~~~

The local source TAR may be streamed twice (~8.9GB
compressed archive) but only up to 20 explicitly labeled
publisher TRAIN videos are extracted to a new ignored root.
Video files remain local. Pose traces, cycle timestamps,
per-video diagnostics and clip names are not included in
the aggregate report; the report is under ignored
data/processed and refuses overwriting.

## Inspect aggregate M43 results

~~~bash
python - <<'PY'
import json
from pathlib import Path

p = Path("data/processed/m43_train_temporal_error_audit.json")
r = json.loads(p.read_text())
print("Split:", r["source_split"])
print("Videos evaluated/excluded:", r["videos_evaluated"],
      r["videos_excluded"])
print("Annotated GT reps:", r["annotated_reps"])
for cond, d in r["m43_diagnostics"]["conditions"].items():
    print("\n", cond, "TP/FP/FN:",
          d["event_tp"], d["event_fp"], d["event_fn"])
    print("FN descriptive categories:",
          d["unmatched_gt_window_observations"])
    print("FP timing categories:",
          d["unmatched_predictions_by_tolerance"])
    print("GT windows with <4 / <6 usable angles:",
          d["all_gt_windows_usable_lt4"],
          d["all_gt_windows_usable_lt6"])
    print("GT windows with low then high:",
          d["all_gt_windows_with_low_then_high"])
    print("Pending side changes / gaps:",
          d["side_switch_resets_while_pending"],
          d["tracking_gap_resets_while_pending"])
print("\nPaired annotated-event recovery:",
      r["m43_diagnostics"]["paired_gt_outcomes"])
PY
~~~

## Research integrity

M43 is a **post-hoc exploratory failure analysis**, not a
new detector. It cannot prove causes of misses, source
frame annotation parity, independent camera-person
generalization or physical squat correctness. We must
not retrofit the already observed three validation
videos as fresh held-out data. The published TEST split
remains untouched until a later model family, selection
rule, exact evaluation protocol, uncertainty unit and
all parameters have been frozen using only legitimate
development data, with cross-subject leakage caveats.
