# M41: TRAIN-only paired 200ms versus 400ms pose observation cadence

## Evidence motivating this study

Owner-run M40 selected 20 annotated RepCount-A Squat TRAIN videos by the
predeclared SHA256 filename-hash policy, with 312 source reps and zero
excluded videos. Frozen M34: TP=14, FP=0, FN=298, F1=0.08589, video
count MAE=14.9. Frozen M39 low-to-high: TP=102, FP=6, FN=210,
precision=94.44%, recall=32.69%, F1=0.48571, count MAE=10.4.
Always-zero control: count MAE=15.6.

The previous M39 pilot on 5 TRAIN videos used archive-order sampling;
its F1=65.26% is NOT a paired estimate of the effect of sample size,
because cohorts differ. None of this is independent TEST performance.

## Frozen M41 experiment

M41 measures whether more time-ordered pose observations affect the
existing M39 low-to-high rule without changing any thresholds.

Each original publisher TRAIN Squat video is decoded **once** with
MediaPipe Solutions Python at nominal 200ms intervals (stride
round(fps * 0.2)). The program compares two streams derived from
that SAME pose-estimation run:

1. dense_200ms: every measured 200ms-target pose sample.
2. subsample_400ms: every other measured sample, indices 0,2,4,...

Both are replayed through exactly the same M39 low-to-high detector,
the original unchanged browser M34 and an always-zero control.
RepCount GT event matching remains one-to-one with the original
M37 time tolerance. Source selection remains fixed SHA256 filename
hash, seed 2026, maximum 20 publisher TRAIN clips. Validation and
TEST annotations and videos are never used.

**Critical experimental limitations:**
The 400ms condition is a subset of poses estimated with the denser
200ms MediaPipe tracker. It is NOT a separate 400ms MediaPipe run,
so differences from the previous M40 baseline cannot be causally
assigned solely to temporal sampling. The M34 internal 400ms sample
gate is NOT reduced: M34 discards dense arrivals less than 400ms
after its previous accepted tick. This experiment explores temporal
observability and should not be presented as an isolated M34
sample-rate intervention.

As with M40, author archive provenance, publisher frame/time
alignment, subject-disjointness, browser/Python parity and external
generalization remain unverified. No exercise form/safety claim
or success threshold is introduced.

## Run on Ubuntu

From the repository root:

~~~bash
git pull --ff-only
source .venv-m20/bin/activate
python -m pytest tests/test_m41_repcount_train_cadence.py -q
npm test --prefix web

python -m scripts.m41_repcount_train_cadence \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --max-videos 20 \
  --root data/external/m41_train_cadence \
  --report data/processed/m41_train_cadence_report.json
~~~

This reads the original ~8.9GB archive twice in streaming mode
(CSV and selected videos), never unpacks all source media, and
extracts at most 20 Squat TRAIN clips to a NEW ignored local root.
All raw landmarks and per-video comparison details are temporary.
The report is aggregate-only and refuses overwriting existing
reports. No external downloads, manual labeling or camera needed.

After completion:

~~~bash
python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path("data/processed/m41_train_cadence_report.json").read_text())
print("TRAIN videos:", r["videos_evaluated"], "excluded:", r["videos_excluded"])
for cond, block in r["conditions"].items():
    print("\n", cond, "frame samples:", block["input_frame_samples_total"])
    for method, m in block["methods"].items():
        print(method, "TP/FP/FN:", m["event_tp"], m["event_fp"], m["event_fn"],
              "P/R/F1:", m["event_precision"], m["event_recall"], m["event_f1"],
              "MAE:", m["mean_absolute_count_error"],
              "zero-rep denominator:", m["zero_rep_negative_control_videos"])
        if "paired_count_error" in m:
            print("Denser better/tied/worse:", m["paired_count_error"])
PY
~~~

In interpreting results, examine TP, FP, recall, precision, count
MAE, actual negative-control denominator, tracking coverage and
paired per-video count-error wins/ties/losses, not F1 alone.
No independent test is warranted until all future method choices
and protocols have been frozen using development-only data.
