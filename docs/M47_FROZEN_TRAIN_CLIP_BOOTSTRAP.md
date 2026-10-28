# M47: Freeze the TRAIN replication and measure paired-video uncertainty

## Confirmed owner results (no TEST use)

M46 ran the original frozen M45 relative-excursion detector
and frozen M39 absolute-angle baseline on **20 new publisher
Squat TRAIN video filenames**, SHA256 train source ranks 21-40,
seed 2026. 20/20 evaluated, zero excluded, **209 GT reps**.

| M46 new TRAIN source files | M39 low-to-high | M45 relative excursion |
| --- | ---: | ---: |
| 200ms TP / FP / FN | 117 / 7 / 92 | 129 / 8 / 80 |
| 200ms event F1 | 0.70270 | 0.74566 |
| 200ms video count MAE | 4.45 | 3.80 |
| Shared-tracker 400ms TP / FP / FN | 94 / 6 / 115 | 121 / 6 / 88 |
| Shared-tracker 400ms event F1 | 0.60841 | 0.72024 |
| Shared-tracker 400ms video count MAE | 5.45 | 4.20 |

On 200ms: the relative detector matched 15 publisher GT
events that original M39 missed and lost 3 original matches.
Per-video absolute count error improved/tied/worsened on
9/9/2 clips. FP rose 7 to 8.

On 400ms: M45 recovered 30 previously missed GT events
and lost 3 previous GT matches; video MAE improved/tied/
worsened on 9/8/3 clips, with FP **unchanged** at 6.

M46 was **not the 20 TRAIN videos used to develop the
relative hypothesis**, so the unchanged M45 rule's gain
replicates on new TRAIN source filenames. No participant,
session, camera or collection-site independence is
proven: multiple files may contain the same person.

M45 M39 200ms original development cohort (first 20 hash
ranks): TP/FP/FN=149/11/163 versus M45 187/15/125,
F1=0.6314 versus 0.7276, MAE 7.90 versus 5.80.
Both cohorts are TRAIN, not publisher TEST.

## M47: uncertainty without another model tweak

M47 does NOT introduce a new detector, model, threshold,
loss function, or training procedure. It precommits to:
- The exact M46 source cohort, original publisher TRAIN
  SHA256 filename hash ranks 21-40, seed 2026 (20 clips).
- The exact M45 and M39 JavaScript implementations
  already merged at M45; original M34/zero unchanged.
- Original M37 event one-to-one GT-end matching tolerance,
  Python MediaPipe Solutions 200ms pose extraction and
  400ms every-other subsampling on the shared 200ms track.
- An existing immutable owner-local
  data/processed/m46_train_frozen_replication.json
  as a frozen reference to which **all recomputed aggregate
  method scores must agree exactly**. M47 saves its SHA256.
- No publisher validation or TEST video, even for negative
  controls; no original model deployment.

To obtain video-level error uncertainty without retaining
per-video identities, M47 evaluates the exact twenty
original M46 TRAIN videos again. Each temporary video
record contains BOTH M39 and M45 stats. In memory only,
resample **entire VIDEO PAIRS**, with replacement, twenty
per draw, using a fixed Python seed 2026 and **5000**
draws. Use the **same sampled clip indices** for both
methods; never bootstrap GT reps as though independent.

For each of the two original M41 cadence conditions,
record the observed candidate-minus-original difference
and simple 2.5th/97.5th **percentile bootstrap interval**
for three measures:

- Micro event F1 across the sampled clips, computed as
  2*TP/(2*TP+FP+FN), including zero-prediction-positive-GT
  samples as 0 for the sole purpose of this statistic.
- Video-mean absolute count error (MAE). **Negative**
  candidate-minus-original MAE means the candidate is better.
- False positive **events per video**. **Positive**
  candidate-minus-original FP rate means the candidate
  causes more false unmatched events.

Bootstrap positive/zero/negative draw **counts** are shown
for transparency; do not interpret these as calibrated
p-values or as the probability of superiority for future
unseen people.

A 95% clip bootstrap percentile interval on twenty clips
**is not a confidence interval for all athletes** when the
same athlete, gym, viewpoint or session may appear in
multiple files. The model-selection history (M43-M45)
makes causal or confirmatory interpretation even harder.
The effective independent sample size could be lower
than 20. Do not infer significant superiority, certification,
physical squat quality or injury safety from this audit.

There are **zero publisher-annotated Squat TRAIN videos
with count 0**. Thus negative-control specificity remains
unavailable despite the FP events on positive-count videos.

## Ubuntu owner-run steps

From the cloned repo:

~~~bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m47_train_clip_bootstrap.py -q

python -m scripts.m47_train_clip_bootstrap \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --reference-report data/processed/m46_train_frozen_replication.json \
  --root data/external/m47_train_bootstrap \
  --report data/processed/m47_train_replication_bootstrap.json
~~~

M47 needs the owner's original unchanged M46 JSON. It
will NOT silently compare a different cohort with that
report. If publisher source hash, rank, rep count, M39/M45
micro event totals or MAE differ on replay, it stops
without creating a bootstrap report. The original M46
report is never modified, and M47 refuses to overwrite
an existing M47 report.

As in M46, TAR.GZ streaming may read the ~8.9GB
archive twice, but only the frozen 20 TRAIN clips are
selected. Selected videos stay in local ignored
data/external/, and temporary pose traces and video
records are deleted. Persistent aggregate JSON goes to
Git-ignored data/processed/; no clips, raw landmarks,
source IDs or per-video metrics are saved.

## Inspect uncertainty

~~~bash
python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path(
    "data/processed/m47_train_replication_bootstrap.json"
).read_text())
print("TRAIN:", r["source_split"])
print("Video N:", r["frozen_source"]["videos_evaluated"])
print("GT reps:", r["frozen_source"]["annotated_gt_reps"])
print("M46 reference SHA256:", r["original_m46_report_sha256"])
print("Bootstrap:", r["analysis_settings"])
for cond, measures in r["paired_clip_bootstrap"].items():
    print("\n", cond)
    for metric, entry in measures.items():
        print(metric, "relative - M39:",
              entry["observed_relative_minus_low_to_high"],
              "95% percentile interval:",
              entry["percentile_95_interval"],
              "draws +/-/0:",
              entry["bootstrap_draws_strictly_above_zero"],
              entry["bootstrap_draws_strictly_below_zero"],
              entry["bootstrap_draws_equal_to_zero"])
PY
~~~

## After M47

If uncertainty remains high, report its width plainly,
do not optimize against the new cohort. An actual final
held-out evaluation would require a single locked
protocol, proper dataset usage and provenance, analysis
of subject/session overlap and an authorized TEST phase
used only after all remaining decisions are frozen.
The three previously examined publisher validation
clips are not an untouched held-out sample.

Future paper-quality work would also need camera
robustness, negative-rep controls from a legitimately
labeled independent development source, participant
stratification and a documented error-bar unit. M47
intentionally does none of these tasks using TEST.
