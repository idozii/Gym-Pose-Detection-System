# M45: Experimental relative-excursion counting versus frozen low-to-high

## Owner-run motivation from M44 (all numbers TRAIN-only)

The owner ran M44 on 20 publisher RepCount-A Squat TRAIN videos
using the same SHA256-selected seed 2026, 312 publisher annotated
reps, 0 excluded videos.

Frozen M39 low-to-high at the shared MediaPipe 200ms-target cadence:
TP 149, FP 11, FN 163, event F1 63.14%, count MAE 7.9.
Of 163 false negatives, 132 showed usable knee angles but no
measured angle <=120 degrees **inside** the publisher GT interval.

Minimum in-GT knee angles among these 132 FN:

| Measured 2D knee angle | FN |
| --- | ---: |
| (120,135] degrees | 46 |
| (135,150] degrees | 34 |
| (150,165] degrees | 23 |
| >165 degrees | 29 |

Thus 80/132 such FN had a minimum within (120,150], but
this statistic is NOT a theoretical upper bound on recoverable
events. The measurement has perspective and pose-estimation
uncertainty, and GT windows may omit part of an angle cycle.

Within fixed +/-750ms publisher GT boundary context, 10/132
showed a low <=120 only before, 7 only after, 4 on both
sides and 111 on neither. Nearly all had usable context
observations (131 before, 130 after).

Of 11 unmatched 200ms predictions, M44 found 8 inside some
publisher GT interval and 3 outside all publisher GT intervals,
although all 11 were outside any GT **end matching tolerance**.
This distinguishes end-timestamp alignment from overlap with
annotated movement. It does NOT establish annotation mistakes.

In the source TRAIN CSV there are no author-labeled Squat
zero-rep videos. Therefore **negative-video specificity remains
unmeasured**. No validation/TEST source can be used to fill this gap.

## One predeclared experimental detector

M45 implements ONE standalone, non-deployed, training-only
relative-excursion comparator motivated **post hoc** by the
same M43/M44 TRAIN diagnostics. It is NOT an independently
validated algorithm or a deployed user-facing counter.

For a given chronological pose sequence:

1. Observe a valid knee-angle measurement >=155 degrees
   on one visible side; track the largest such angle as peak.
2. Observe a >=35 degree drop from that peak, on the same
   tracking side, and remember a local trough.
3. Count one candidate event if after at least 400ms and
   within 12,000ms, the angle rises >=35 degrees from the
   trough AND ends >=155 degrees on the same side.
4. Reset incomplete attempts after >1250ms tracking gap,
   a visible-side change or a pending-return timeout.
5. After each complete return, use that high angle as the
   next peak anchor. Never count multiple events from
   simply remaining standing.

No absolute <=120 degree low-pose requirement is applied.
All constants are frozen in M45 tests and report metadata:
155 degree high, 35 degree drop and recovery, 400ms minimum
return, 12,000ms timeout, 1250ms tracking gap.
The 35 degree value comes from the original M34/M39 minimum
excursion; high and timing bounds also reuse frozen values.
There is no GT-informed inference or per-video threshold
search, and no new model training.

This detector will inherently be conservative for source
views never showing a >=155 degree high posture; it is NOT
expected to solve all FN including very high in-GT minima.

## Exact controlled comparison

For each of up to 20 identical fixed-hash publisher TRAIN
Squat clips, extract the original Python MediaPipe 200ms
pose sequence ONCE, also replay every other sample as
the shared-tracker ~400ms control, as in M41-M44.

Evaluate the original, UNCHANGED:
- Always-zero count baseline.
- M34 RepCycleCounter as shipped in the website (400ms gate).
- M39 absolute <=120 and >=155 low-to-high comparator.
- One M45 relative-excursion candidate.

Use the exact original M37 GT-end one-to-one matching and
tolerances for ALL event comparisons. Compare event TP,
FP, FN, micro precision/recall/F1, video count MAE,
signed bias and exact/within-one error. **Do not select
a model by F1 alone.**

In addition, M45 counts GT rep matches paired between
the original low-to-high and relative candidate:
matched by both, new-only, original-only, neither.
Report candidate unmatched prediction position inside/
outside original publisher GT intervals, and candidate
incomplete-cycle reset counters. Report how many video
count errors improved, tied, or worsened.

The M45 runner also recomputes and validates unchanged
M41, M43 and M44 results using the same input trace.
Real-video scores are never asserted by the release itself.

## Reproducibility on Ubuntu

From the repository root:

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m45_train_relative_motion_benchmark.py -q
npm test --prefix web

python -m scripts.m45_train_relative_motion_benchmark \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --max-videos 20 \
  --root data/external/m45_train_relative_motion \
  --report data/processed/m45_train_relative_motion_report.json
~~~

The original locally authorized ~8.9GB TAR.GZ is read
in streaming mode for author TRAIN annotations and
up to twenty exact TRAIN videos. All extracted video
files remain in ignored local external directories.
Pose traces, video IDs, event times and per-video
scores are temporary, and are NOT persisted in the
aggregate report. Prior M39-M44 reports are not overwritten.

The analysis runs entirely in existing local Python
and Node.js tooling, with no dataset rehosting,
camera capture or external training endpoint.

## View the experiment outcome

~~~bash
python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path(
    'data/processed/m45_train_relative_motion_report.json'
).read_text())
print("Split:",r["source_split"])
print("N evaluated / excluded:",r["videos_evaluated"],r["videos_excluded"])
print("GT reps:",r["annotated_reps"])
print("Source TRAIN CSV:",r["publisher_train_annotation_sha256"])
for cond,block in r["conditions"].items():
    print("\n",cond)
    for name,metric in block["methods"].items():
        print(name,"TP/FP/FN:",
              metric["event_tp"],metric["event_fp"],metric["event_fn"],
              "F1:",metric["event_f1"],
              "Count MAE:",metric["mean_absolute_count_error"],
              "Signed bias:",metric["mean_signed_count_error"])
    print("Paired GT matches:",
          block["paired_diagnostics"]["paired_gt_vs_original_low_to_high"])
    print("Video count error change:",
          block["paired_diagnostics"]["paired_video_absolute_count_error"])
    print("Candidate unmatched FP inside/outside GT:",
          block["paired_diagnostics"]["candidate_fp_location"])
PY
~~~

## Reporting and honest scientific status

This algorithm was designed AFTER viewing M43 and M44
outcomes for the very same TRAIN cohort. Those source
videos are **development data**; even higher F1 on these
clips is a post-hoc training-set improvement, not a
confirmation of generalization, causal superiority or
a paper-ready benchmark result.

Fewer matching errors can coexist with more false
events, and detecting something during a source GT
interval does not automatically make it a correct
event under the fixed one-to-one matching policy.

No threshold tuning, held-out TEST use, official
source provenance verification, independent participant
sampling, temporal annotation alignment audit or
negative control validation was performed here.
The next research stage should set up controlled TRAIN
model selection and freeze a *single* model and protocol
before one independent TEST evaluation. Do not call
the previously examined validation samples independent.
