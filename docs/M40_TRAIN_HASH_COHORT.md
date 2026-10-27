# M40: Larger TRAIN-only counting baseline with fixed hash cohort

## Owner-run M39 exploratory result (28 October 2026)

The original five publisher TRAIN Squat videos contain 56 source annotated
reps; all five were decoded and evaluated successfully.

| Method | TP | FP | FN | Event F1 | Video count MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always-zero | 0 | 0 | 56 | Undefined | 11.2 |
| Original M34 | 2 | 0 | 54 | 0.0690 | 10.8 |
| Fixed low-to-high | 31 | 8 | 25 | 0.6526 | 3.8 |

M39 low-to-high precision: 31/39 = 79.49%.
M39 low-to-high recall: 31/56 = 55.36%.
Compared with original M34, video count MAE declined by 7.0
reps/video (64.8%) on the same FIVE TRAIN clips.

These are **development-only results**, NOT statistically significant
or generalizable held-out test performance, and NOT proof of injury
safety, exercise correctness, browser/Python estimator parity or
source annotation/video synchronization. Low-to-high made eight
false-positive events; this failure mode cannot be ignored.

## Why M40

The M39 convenience cohort used the first five matching files in
the source TAR member order; those files are not a random or
representative sample. M40 predeclares a filename-based fixed
hash sampling policy, so the chosen source filenames do not depend
on archive member order, detector outcomes or performance scores.

With `--selection-policy stable_hash --selection-seed 2026`,
the extractor first reads publisher TRAIN annotations and sorts
Squat source basenames by:
`SHA256("repcount-m40-train-v1:2026:" + lowercase_basename)`.
Only up to 20 entries are retained. When extracting, an original
TAR member must have an explicit `train` or `training` directory,
and must match one of the selected TRAIN CSV basenames.
No validation/test clips, manually relabeled videos or webcam footage
are decoded. Old `archive_order` remains the M39 default.

This is deterministic clip-level sampling and still does NOT prove
subject-, view- or scene-disjointness. Selecting a different cohort
does not retroactively make the previously observed 5 TRAIN clips
an independent test set. We should compare method behavior on
the larger TRAIN cohort, not train or tune on source TEST data.

## Reproduce on Ubuntu

From the project repo root:

```bash
git pull --ff-only
source .venv-m20/bin/activate

python -m pytest tests/test_m39_repcount_train_baselines.py tests/test_m37_repcount_tar.py -q

python -m scripts.m39_repcount_train_baselines \
  --source-archive ~/Downloads/RepCountA.tar.gz \
  --selection-policy stable_hash \
  --selection-seed 2026 \
  --max-videos 20 \
  --root data/external/m40_train_hash20 \
  --report data/processed/m40_train_hash20_baselines.json
```

The original 8.9 GB archive is streamed twice (publisher CSV and
selected videos). It is NOT entirely extracted. Video files and
temporary poses remain under ignored local data folders, and no
additional download, hosting subscription or manual annotation is
required. The command will not overwrite a report or extracted clip.

Only aggregated TP/FP/FN, F1, precision/recall, session MAE, signed
count bias, negative-control errors, valid-pose coverage and exclusion
categories are retained. The output records source split, annotation
SHA256, sampling policy, selection seed and actual video denominator,
not individual names, cycle timestamps, source video frames or poses.

View the output with:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('data/processed/m40_train_hash20_baselines.json')
r = json.loads(p.read_text())
print("Split:", r['source_split'])
print("Policy/seed:", r['selection_policy'], r['selection_seed'])
print("Video denominator:", r['evaluated_videos'], "excluded:", r['excluded_videos'])
for name, m in r['results'].items():
    print(name,
          "TP/FP/FN:", m['event_tp'], m['event_fp'], m['event_fn'],
          "F1:", m['event_f1'],
          "MAE:", m['mean_absolute_count_error'],
          "negative clips with FP:", m['zero_rep_negative_controls_with_false_cycles'])
PY
```

When fewer than 20 publisher TRAIN Squat videos match, report the
actual evaluated N, exclusions and reason. The selection rule does
not silently add next-best or high-scoring alternatives.

## Scientific next gate

Do not tune thresholds against the three already-analyzed validation
clips or use publisher TEST for exploratory model selection. Examine
false positives and zero-rep negative controls on TRAIN, verify
source annotation/frame synchronization if possible, and develop
a frozen subject-aware experiment protocol before one final
independent TEST assessment. The same-split TRAIN results are
development evidence only, and we cannot compute defensible
population uncertainty from the initial five clips.
