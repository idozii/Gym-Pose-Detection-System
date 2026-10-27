# M39: Fixed temporal return baseline on publisher TRAIN only

## Why this experiment

The M38 three-video *validation* failure audit succeeded, with
1/45 M34 detected events (F1 0.04348), 14.67 count MAE,
vs 15.0 count MAE for the always-zero control. In M37.2,
44/45 publisher cycle intervals had fewer than six usable
knee-angle samples. These findings are **exploratory**.
The three previously inspected validation clips are no longer
untouched data and cannot be used to select M39 settings.

M39 starts a separate **development-only** comparison on
publisher TRAIN annotations and clips from the user's legally
obtained local RepCountA.tar.gz. It does not access publisher
validation or TEST. The authors' CSV must be embedded at
`LLSP/annotation/train.csv`; video members must have an
explicit train/training directory. Splitless video paths are
rejected for M39 to avoid accidental cross-split inclusion.

## Frozen methods before reading TRAIN results

All methods consume the **same** 400ms-target chronological
MediaPipe Solutions Python landmark trace from the selected
TRAIN videos, with unchanged source one-to-one event matching:

- **Always zero:** predict no positive event, count=0.
  Event precision and F1 are `null` when no event is
  predicted. Per-video count MAE is still meaningful.
- **Original M34:** call the unchanged browser
  `RepCycleCounter` through the original M37 evaluator.
  M34 requires confirmed high, low and return observations.
- **Low-to-high return control (M39):** after a measured
  knee angle <=120 degrees, count on the first later
  >=155 degree observation of the same limb with at least
  35 degree amplitude and elapsed time 400ms to 12s.
  Discard unfinished motion on a >1250ms tracking gap
  or a LEFT/RIGHT switch. Do not count again until another
  low observation. These geometric angle thresholds, gap
  and maximum duration reuse M34's published constants;
  the new single-observation rule is a **deliberately
  simpler exploratory comparator**, not an established
  robust detector or trained model.

The candidate is motivated by earlier post-hoc validation
error analysis, so exploratory TRAIN results are **not**
independent replication. No threshold grid search,
data-dependent setting selection, cross-validation or
test access is implemented here.

## Run on Ubuntu

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m39_repcount_train_baselines.py -q
npm test --prefix web

python -m scripts.m39_repcount_train_baselines \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --max-videos 5 \
  --root data/external/m39_train_only \
  --report data/processed/m39_train_only_baseline_results.json
```

The TAR.GZ reader iterates the compressed author archive to
discover embedded publisher TRAIN labels, and again to
copy **at most five** explicitly labeled TRAIN Squat clips.
It may read the ~8.9GB archive sequentially twice;
it never unpacks the full archive. The training source
remains outside Git and retains its unverified independent
provenance and source-frame sync limitations.

The default `--max-videos 5` is a resource-bounded
smoke study, **not** a representative random sample.
All source selections are deterministic archive order.
Raw landmark traces and per-video method outputs are
ephemeral temporary files, removed after evaluation.
Only cohort-level totals, event TP/FP/FN, precision,
recall, F1, video count MAE, bias, exclusion categories
and detection coverage are kept under git-ignored
`data/processed/`.

No manually labeled CSV, browser camera, cloud inference,
new data download or paid hosting is needed. Existing
extracted clips are never silently overwritten; for a
second run supply new `--root` and `--report` paths.

## Interpretation and next research gates

A higher TRAIN F1 for low-to-high would show that a simpler
temporal rule can detect more publisher events **on a
development subset**; it could also increase spurious reps.
Report both recall and FP/precision and zero-rep control
counts. An FP-heavy candidate is **not** automatically better.

This is an ablation-style exploratory development comparison
on a very small set, not a final research achievement.
Before a TEST evaluation, scale TRAIN analysis using
legitimate accessible source data, settle the temporal
model and thresholds using training/development data only,
freeze methods, metadata and matching protocol, check
participant/scenario leakage and camera-view confounds,
and make one independently held-out publisher TEST
assessment if allowed. Do not infer good/bad exercise
form or injury risk.
