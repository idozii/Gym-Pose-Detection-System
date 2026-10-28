# M48: Freeze M46/M47 TRAIN evidence before any TEST evaluation

## Owner-observed evidence from 20 new publisher TRAIN videos

M46 evaluated 20/20 publisher TRAIN Squat videos, original
filename-SHA256 source ranks 21–40 (seed 2026), with 209
annotated events and zero exclusions. The same M45 relative
excursion detector and M39 absolute-angle control were unchanged.

| Result | M39 | M45 |
| --- | ---: | ---: |
| 200ms TP/FP/FN | 117/7/92 | 129/8/80 |
| 200ms micro F1 | 0.70270 | 0.74566 |
| 200ms video count MAE | 4.45 | 3.80 |
| 400ms paired-subsample TP/FP/FN | 94/6/115 | 121/6/88 |
| 400ms micro F1 | 0.60841 | 0.72024 |
| 400ms video count MAE | 5.45 | 4.20 |

M47 replayed these exact 20 source videos and computed 5000
seeded paired video bootstrap draws:

- 200ms M45−M39 F1 +0.042962, 95% percentile [0.001847,0.084923];
  video count MAE −0.65, [-1.25,−0.10]; FP events/video +0.05,
  [−0.20,+0.30].
- 400ms M45−M39 F1 +0.111824, [0.031327,0.190676];
  video count MAE −1.25, [−2.45,−0.20]; FP events/video 0,
  [−0.20,+0.25].

These are **TRAIN-only exploratory percentile intervals**,
conditional on a source-video sample of 20. Distinct filenames
do not imply different people, sessions or viewpoints. Bootstrap
draws are not new independent examples or calibrated p-values.
The FP interval contains possible increases; no evidence of
non-inferiority or zero false alarms on negative-only clips.
The original publisher Squat TRAIN annotation lacks zero-rep
negative controls.

## What the code freezes

M48 takes the ORIGINAL locally stored M46 and M47 JSON reports,
without reading any video, validation or TEST data. Before
writing a local SHA256 manifest, it strictly checks:

1. Both report protocols and exact original TRAIN rank offset
   20 / seed 2026 / 20 videos / 209 ground-truth repetitions.
2. M47's original-M46 report SHA256 matches the original
   M46 file **byte-for-byte**, and both reports have the
   same original publisher TRAIN annotation SHA256.
3. Owner-observed 200ms and 400ms M39/M45 TP/FP/FN/F1,
   video count MAE, original frozen algorithm rule and
   the 5000-draw bootstrap delta/bound endpoints match.
4. The methods and original M34 were unchanged, and the
   two source sampling cadences share one MediaPipe
   tracking history, as in M41–M47.

M48 then saves both owner JSON byte SHA256 values, Git HEAD,
the publisher annotation SHA from those reports, and the
SHA256 values of 17 source modules responsible for
selection, pose tracking, M34, M37 GT matching, M39,
M45 and benchmark computation. Only aggregate information
and hashes persist under ignored data/processed.

The lock is an **audit fingerprint, not an official signed
attestation**. Its read-only verification mode checks both
report bytes and all fingerprinted source bytes again.

## Ubuntu commands

After fetching the newly merged main branch:

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git status --short
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m48_freeze_repcount_evidence.py -q

python -m scripts.m48_freeze_repcount_evidence \
  --m46-report data/processed/m46_train_frozen_replication.json \
  --m47-report data/processed/m47_train_replication_bootstrap.json \
  --report data/processed/m48_train_evidence_freeze.json
~~~

Read-only verification, with the exact same original JSON
files and source code:

~~~bash
python -m scripts.m48_freeze_repcount_evidence \
  --verify-freeze \
  --m46-report data/processed/m46_train_frozen_replication.json \
  --m47-report data/processed/m47_train_replication_bootstrap.json \
  --report data/processed/m48_train_evidence_freeze.json
~~~

M48 refuses to overwrite an existing lock. If source code
or report bytes change after freezing, verification fails.
It does not modify the original reports or detector, and
does NOT give any command to access TEST data.

Important: M48 hashes actual local source bytes. Ensure
those research source paths match the pulled main Git HEAD
when creating a provenance-aligned lock. Other uncommitted
files outside the fingerprint list are not part of M48.

## TEST-readiness blockers not resolved by M48

- Official author-provided archive provenance, data usage
  permission and annotation/video frame synchronization
  remain unverified.
- No proof that publisher source files are subject-, session-
  or camera-independent across original splits.
- No publisher Squat TRAIN count-zero controls exist; an
  authorized independent labeled negative set is needed.
- Python MediaPipe video inference is not proof of browser
  MediaPipe/JS runtime parity.
- Three VALIDATION clips were previously inspected in M38;
  they are not untouched held-out validation.
- The final TEST cohort, matching tolerances, primary metrics,
  subject independence checks, exclusions and stopping rule
  must be frozen BEFORE a one-time authorized TEST evaluation.

M48 is a pre-TEST **research integrity** milestone, not a
new model, an independent test result, conference-paper
acceptance, medical safety or squat-form certification.
