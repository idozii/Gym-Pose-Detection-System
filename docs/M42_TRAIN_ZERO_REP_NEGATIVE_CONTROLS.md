# M42: Publisher-annotated zero-repetition TRAIN negative controls

## Why this experiment matters

The owner ran M41 on a frozen set of 20 publisher TRAIN Squat
clips with 312 annotated repetitions and zero exclusions.

| Frozen low-to-high comparison | Subsample 400ms | Dense 200ms |
| --- | ---: | ---: |
| Event TP / FP / FN | 118 / 3 / 194 | 149 / 11 / 163 |
| Event precision | 97.52% | 93.13% |
| Event recall | 37.82% | 47.76% |
| Event F1 | 54.50% | 63.14% |
| Video count MAE | 9.55 | 7.90 |

In the paired count comparison, 200ms reduced absolute count
error on 10 videos, tied on 9, and worsened on 1. M34 made the
same 18 TP / 1 FP at both conditions because its internal
400ms gate remained unchanged.

**The 20 selected source videos had ZERO annotated zero-rep clips.**
The observed 3 versus 11 false events therefore tell us
something about matching errors in *positive* videos, but
nothing about the false-alarm rate on genuinely zero-labeled
source videos. Treating "zero false alarms on zero-rep videos"
with a denominator of zero as evidence of specificity is invalid.

## New M42 negative-control design

M42 adds an opt-in source annotation filter, selecting
**only** RepCount-A Squat TRAIN CSV rows with publisher
count exactly 0 and no L* cycle frame bounds. The filter
runs BEFORE filename-hash selection, video extraction or
inference. It uses the existing fixed SHA256 seed 2026,
requires video members explicitly under train/training
folders, rejects ambiguous split labels and rejects
valid/test video paths. At most twenty negative videos
are extracted. No other exercise or unlabeled video
can be silently relabeled as a zero-rep negative control.

Both frozen low-to-high and original M34, plus always-zero,
are evaluated on the same 200ms-target MediaPipe track
and its every-other 400ms subsample. There are **no**
changes to decision thresholds, original M34 sampling gate
or M37 one-to-one event matcher. This remains a TRAIN-only,
exploratory experiment, not subject-independent TEST.

When all GT source counts are zero, event recall and event
F1 are UNDEFINED (JSON null). Report meaningful outcomes:
- number of zero-labeled TRAIN clips evaluated and excluded;
- total predicted false-positive cycle events;
- zero-GT videos with one or more false-positive events;
- video count MAE (equivalent to mean false cycles per video);
- dense versus subsampled paired per-video count-error changes;
- actual temporal pose sampling and coverage.

An annotation marked zero does not prove the subject
stood still or never moved. It means the publisher
recorded no countable reps under their annotation rule.
All original archive provenance, person/view coverage,
source frame alignment, browser/Python parity and
false-positive generalization remain unverified.

## Run on Ubuntu

From your existing repository root:

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m42_repcount_negative_controls.py -q

python -m scripts.m41_repcount_train_cadence \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --cohort-filter zero \
  --max-videos 20 \
  --root data/external/m42_train_zero_reps \
  --report data/processed/m42_train_zero_rep_controls.json
~~~

M42 reuses the proven M41 runner. It reads the locally
authorized TAR.GZ in streaming mode, potentially decompressing
the ~8.9GB source twice, but extracts only chosen TRAIN
negative-control videos. Original M40/M41 reports remain
untouched; reports and extracted videos are Git-ignored.

**If publisher TRAIN annotations contain no labeled zero-rep
Squat rows**, the command stops with an explicit error
before extracting videos and writes no accuracy report.
In this case, do NOT substitute zero-rep videos from
validation or TEST or assume specificity. Find another
independently annotated, authorized development negative
cohort first.

## Read the negative-control result

~~~bash
python - <<'PY'
import json
from pathlib import Path
p = Path("data/processed/m42_train_zero_rep_controls.json")
r = json.loads(p.read_text())
print("Split:", r["source_split"])
print("Filter:", r["source_selection"]["cohort_filter"])
print("Selected/evaluated/excluded:",
      r["videos_selected"], r["videos_evaluated"],
      r["videos_excluded"])
print("GT events:", r["annotated_reps"])
for cond, block in r["conditions"].items():
    print("\n", cond)
    for name, m in block["methods"].items():
        print(name,
              "FP events:", m["event_fp"],
              "False-alarm videos:",
              m["zero_rep_negative_controls_with_false_cycles"],
              "Negative videos:", m["zero_rep_negative_control_videos"],
              "Mean false cycles/video:", m["mean_absolute_count_error"])
PY
~~~

No browser webcam, fresh labels, new external data download,
training, hyperparameter search, personal video uploads or
original TEST access. Do not advertise paper-ready
performance from these development-only controls.
