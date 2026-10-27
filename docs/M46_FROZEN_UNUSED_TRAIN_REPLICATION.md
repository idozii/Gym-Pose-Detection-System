# M46: Frozen M45 replication on unused publisher TRAIN source clips

## Why freeze now?

The owner ran M45 on **the same 20 publisher TRAIN Squat clips** used
to motivate M43-M44. The observations are promising but remain in-sample
development results:

| Metric at 200ms | Frozen M39 low-to-high | M45 relative excursion |
| --- | ---: | ---: |
| True positive GT matches | 149 | 187 |
| False positive event predictions | 11 | 15 |
| Missed GT events | 163 | 125 |
| Micro event F1 | 0.63136 | 0.72763 |
| Video count MAE | 7.90 | 5.80 |

M45 gained 41 GT-event matches M39 lacked, but lost 3
previous M39 matches. 122/312 annotated GT events were
missed by **both**. Video count error improved on 6/20
videos, was unchanged on 12/20 and worsened on 2/20.
False positives increased from 11 to 15.

At shared-tracker every-other 400ms: M39 TP/FP/FN
118/3/194, event F1 0.54503, count MAE 9.55;
M45 TP/FP/FN 155/8/157, event F1 0.65263,
count MAE 7.45. Per-video errors improved in 7,
tied in 12, worsened in 1.

**M45's architecture was developed after observing its original
20-video TRAIN cohort.** Those metrics cannot establish a
replicable improvement and are not TEST accuracy. Rather than
modifying parameters to improve the same sample, M46 FREEZES
both the original M39 rule and original M45 candidate.

## Prespecified M46 source cohort

The original M40 source selector ranks filenames in publisher
TRAIN Squat annotations using:
SHA256("repcount-m40-train-v1:2026:" + lowercase source filename).

Previously observed M40-M45 results used the first 20 ranks.
M46 selects ONLY ranks **21-40** (zero-index Python slice
[20:40]), in their deterministic hash order, independently of
the archive TAR member order or outcome scores.

The new source clips have filenames disjoint from all of the
original 20 clips by construction. No validation or TEST
source or annotations are accessed. The same participant,
scene or view might still appear under another TRAIN filename:
**subject-, session-, and view-level independence are not
established**.

- M46 never decodes, annotates, trains on, or selects any
  clips from publisher validation/TEST.
- If fewer than 40 publisher TRAIN Squat filenames exist,
  use all eligible post-first-20 ranks and disclose the actual
  N. Do **not** add first-20 source files as replacement.
- If no TRAIN Squat filenames exist after rank 20,
  stop before decoding any video or creating a report.
- If the archive lacks any selected TRAIN video member,
  stop instead of filling the cohort with another source clip.
- If a selected clip fails pose inference or evaluation,
  report the actual exclusion count and error type; do not
  cherry-pick an alternate video. Prefer interpreting full
  20/20 results and report coverage explicitly.
- Source selection rule and detector parameters are frozen
  before observing any M46 owner-run real video results.

## Exactly unchanged methods

M46 is **NOT a new detector**. It invokes the original
M45 JS relative-excursion code, with exact original
155-degree high anchor, >=35-degree same-side excursion,
400-12,000ms temporal return, <=1250ms pose tracking gaps,
and the original M39 absolute low-to-high comparator,
M34 and always-zero baselines.

It reuses original M41 200ms Python MediaPipe Solutions
landmarks, every-other subsample as the shared tracker
400ms observation, M43/M44 diagnostics and M37 one-to-one
GT-end event matching/tolerance. It does not lower or
tune absolute low/high angles, add training, change
publisher GT windows, recalibrate source timestamps,
alter deployed browser JS, or claim physical squat safety.

## Predeclared interpretation

Primary descriptive outcome: whether, on this NEW TRAIN
source-file cohort, original M45 relative excursion has
higher event F1 and lower video count MAE than unchanged M39
at **dense_200ms**, without an unacceptable rise in FP.
Assess both outcomes; do not focus on F1 alone.

Secondary reports: precision/recall, TP/FP/FN,
signed count bias, GT paired matches recovered/lost by
M45 vs M39, per-video count error wins/ties/losses, and
shared-tracker subsample_400ms sensitivity.
The report includes source TRAIN CSV SHA256, fixed seed,
rank offset 20, source-level disjointness flag and actual
selected/evaluated/excluded denominators; only aggregates
are persisted, not filenames or per-video predictions.

No automatic go/no-go or new thresholds are introduced.
This *fresh TRAIN source-file replication* improves internal
development credibility but is NOT fully independent
validation, subject-disjoint holdout, or published TEST accuracy.
The original TRAIN Squat CSV contains no zero-rep clip,
so negative control specificity remains unmeasured.

## Ubuntu commands

From the existing project repository root:

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m46_train_unseen_cohort.py -q
npm test --prefix web

python -m scripts.m45_train_relative_motion_benchmark \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --max-videos 20 \
  --cohort-offset 20 \
  --root data/external/m46_train_rank21_40 \
  --report data/processed/m46_train_frozen_replication.json
~~~

The M45 runner and JavaScript implementation are intentionally
reused verbatim. Offset=0 remains the exact original M45
archive selection and default. Offset=20 selects the
non-overlapping TRAIN ranks 21-40, with no data-dependent
alternatives. Output folders are different from the
existing M45 result; the report refuses overwrite.
Source archive scanning remains streaming, and any
extracted video/temporary pose data stays Git-ignored
under the owner's local files.

## Review result

~~~bash
python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path(
    "data/processed/m46_train_frozen_replication.json"
).read_text())
print("TRAIN:", r["source_split"])
print("Rank offset:", r["source_selection"]["source_hash_rank_offset"])
print("Name-level disjoint:",
      r["source_selection"]["source_clip_filename_disjoint_from_m40_m45"])
print("N evaluated / excluded:", r["videos_evaluated"], r["videos_excluded"])
print("GT events:", r["annotated_reps"])
for cond, block in r["conditions"].items():
    print("\n", cond)
    for name in ("low_to_high", "relative_excursion"):
        m = block["methods"][name]
        print(name, "TP/FP/FN", m["event_tp"], m["event_fp"], m["event_fn"],
              "F1", m["event_f1"],
              "MAE", m["mean_absolute_count_error"])
    p = block["paired_diagnostics"]
    print("paired GT:", p["paired_gt_vs_original_low_to_high"])
    print("video error better/tied/worse:",
          p["paired_video_absolute_count_error"])
    print("candidate FP positions:", p["candidate_fp_location"])
PY
~~~

Do not adjust M45's source or thresholds after seeing this
replication's scores and call the same data independent
confirmation. If the improvement does not replicate,
record the failure plainly. Before one final independent
publisher TEST evaluation, separately plan participant/
session overlap checks, missing zero-rep negative controls,
annotation/video synchronization, original video licensing,
and a frozen end-to-end evaluation protocol.
