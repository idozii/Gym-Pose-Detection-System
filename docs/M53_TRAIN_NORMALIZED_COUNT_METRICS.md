# M53 - Publisher TRAIN normalized count MAE, OBO and exact-count metrics

## Predeclared scope and motivation

The frozen M46/M47 evaluation tested the unchanged M39 low-to-high
comparator and M45 relative-excursion Squat candidate on 20 original
RepCount-A **TRAIN** source video filenames (hash ranks 21-40).
It was distinct by SOURCE FILENAME from the original 20 development
TRAIN clips (first20). Subject, session and camera independence
were NOT verified, and official TEST remains untouched.

Frozen M46 point estimates at 200ms:
- M39: 117 TP / 7 FP / 92 FN, event F1 0.70270,
  absolute video count MAE 4.45.
- M45: 129 TP / 8 FP / 80 FN, event F1 0.74566,
  absolute video count MAE 3.80.
- GT = 209 annotated Squat reps, 20/20 original TRAIN clips.
- Both original 400ms shared-pose-tracker subsample conditions
  remain unchanged and are also processed.

M53 uses the same 20 per-video pairs to calculate missing
**normalized MAE**, **Off-By-One (OBO)** and **exact-count accuracy**.
The original M39 aggregate code already contains some OBO/exact
fields; M53 recomputes them from each original video count and
cross-checks those fields against M46.

Because individual GT video counts and predictions are NOT in the
aggregate, exact normalized MAE is NOT mathematically recoverable
from 20-video GT totals, event TP/FP/FN or count MAE.
M53 therefore requires the owner's original authorized TRAIN TAR,
the owner's unchanged original M46/M47 reports and M48 lock,
and a new run of the unchanged original video evaluator.

## Metric definitions for positive-GT clips

Let g_i > 0 denote original publisher Squat GT count and p_i
the unmodified predicted cycle count for original video i.

- **Normalized video-mean count MAE:**
  (1/N) sum_i |p_i-g_i| / g_i.
- **OBO (Off-By-One) fraction:**
  (1/N) sum_i 1[|p_i-g_i| <= 1].
- **Exact-count fraction:**
  (1/N) sum_i 1[p_i == g_i].
- **Original absolute video count MAE:**
  (1/N) sum_i |p_i-g_i|.

These are **per-video mean** metrics; normalized MAE
does NOT equal sum(abs errors)/sum(GT) unless special
conditions happen to hold. Positive GT is mandatory for
the normalized metric; publisher TRAIN Squat lacks
annotated zero-rep source clips, so no ad hoc smoothing
or zero-denominator substitutions are introduced.
OBO and exact-count are summary fractions, not
per-event timing correctness.

The normalization is a precise, explicitly labeled
relative-count metric; comparing values to TransRAC or
other papers requires exact dataset split, source,
model input, zero-count handling and original paper
metric definition to match. It is not legitimate to
claim published baseline superiority from M53 alone.

M53 also reports **paired-video percentile 95% intervals**
for M45-M39 differences of the four metrics. Use the
same entire source video paired for both methods in
each bootstrap resample: 5000 draws with fixed seed
2026. This analysis is exploratory on 20 TRAIN
filenames and may understate dependency between
the same participant/session in different files.

## Integrity and immutability

1. Call the unchanged M49 original Git/source gate
   in **check-only** mode. M49 verifies all 17 frozen
   M48 source SHA256 entries, both tracked Git revisions,
   original owner-local M46/M47 report bytes, M47 M46
   reference SHA and TRAIN annotation checksum.
2. Re-run the original M47 evaluator (using its original
   fixed hash offset=20, seed=2026, original M45 Node
   comparator, frozen Python MediaPipe Solutions pose
   tracking and original GT/matcher). M47 REFUSES
   output if any 200ms/400ms M39/M45 aggregate differs
   from the owner M46 reference.
3. Calculate per-video M53 metrics in private memory
   only; cross-check absolute count MAE, OBO fraction,
   exact-count fraction, GT and predicted totals
   against original M46.
4. Use W6 to compare the original and browser M45 JS
   logic on each EXACT SAME Python landmark trace.
   This never opens a browser, does not run Tasks Vision,
   and is explicitly NOT end-to-end browser/Python parity.
5. Save only two Git-ignored summary reports; no
   source filenames, person IDs, per-video scores,
   video, landmarks, source GT frames or event times.
   Private intermediate pose traces are deleted by
   the original M47 TemporaryDirectory. Video source
   extraction stays under ignored data/external.

No validation or official TEST videos, annotations,
or media bytes are selected. The source-archive reader
has split='train' and the unchanged hash source policy.

## Owner Ubuntu command

Run AFTER fetching the merged M53/W6 branch into main:

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git status --short
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m53_w6_train_metrics_parity.py -q
npm test --prefix web

python -m scripts.m53_w6_train_metrics_parity \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --root data/external/m53_train_metric_parity \
  --m46-report data/processed/m46_train_frozen_replication.json \
  --m47-report data/processed/m47_train_replication_bootstrap.json \
  --m48-lock data/processed/m48_train_evidence_freeze.json \
  --report data/processed/m53_train_normalized_count_metrics.json \
  --w6-report data/processed/w6_train_same_landmark_parity.json
~~~

This can involve local TAR stream scanning, MP4 extraction and
20 videos of CPU pose inference. It WILL NOT re-run official TEST.
The two reports refuse overwrite. Do not delete/replace frozen
M46/M47/M48 reports just to rerun M53.

Summarize the owner result:

~~~bash
python - <<'PY'
import json
from pathlib import Path
root=Path("data/processed")
m=json.loads((root/"m53_train_normalized_count_metrics.json").read_text())
w=json.loads((root/"w6_train_same_landmark_parity.json").read_text())
for cond,methods in m["conditions"].items():
    print("\nM53:",cond)
    for model,stats in methods.items():
        print(model,"normalized MAE",stats["mean_normalized_absolute_count_error"],
              "OBO",stats["off_by_one_fraction"],
              "exact",stats["exact_count_fraction"],
              "absolute MAE",stats["absolute_count_mae"])
    print("paired intervals",m["paired_video_percentile_bootstrap"][cond])
for cond,counts in w["conditions"].items():
    print("\nW6:",cond,
          "same input mismatches",counts["direct_same_input_event_disagreements"],
          "gated mismatches",counts["gate_same_input_event_disagreements"],
          "gated dropped frames",counts["gate_dropped_frames"])
PY
~~~

## Interpret honestly

If M45 yields lower normalized count MAE or higher OBO,
report those exact owner-computed numbers; do not predict
or invent metrics before owner execution.
A small difference could be sensitive to source
GT counts; video bootstrap precision might be
optimistic if source participants overlap.
No new M45 detector or tuned threshold is created.
After M53, the next research gate remains source
licensing/provenance, GT semantics, independent
zero-rep negative controls, and frozen one-time
official TEST protocol when legitimately authorized.
