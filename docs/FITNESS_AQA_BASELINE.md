# Fitness-AQA: pose-coverage QA and two-error baseline (train/val only)

This is a separate, error-specific research experiment; it does **not**
replace the binary `correct/incorrect` live webcam baseline.

## Evidence from the owner's 24-video smoke test

| Metric | train | val | test |
| --- | ---: | ---: | ---: |
| Videos processed | 8 | 8 | 8 |
| Sampled frames | 65 | 54 | 69 |
| Annotation boundary rejected | 3 | 7 | 10 |
| Pose accepted | 31 | 40 | 33 |
| Pose rejected | 31 | 7 | 26 |
| Eligible-frame pose coverage | 50.0% | 85.1% | 55.9% |
| Accepted knees-forward positive frames | 10 | 13 | 8 |
| Accepted knees-inward positive frames | 2 | 1 | 3 |

Total: 188 sampled; 20 near annotation boundaries; 104 accepted
of 168 eligible (61.9%). No source video failed to decode.

These counts show **pose-filter coverage differences**, not model accuracy.
Only 2 positive inward-knee training frames survived, so **do not train or
tune on the 8-video pilot**.

## A. Audit your existing pilot right now

No video processing or re-downloading is necessary:

```bash
python -m scripts.audit_fitness_aqa_pose
```

The audit loads CSVs and extraction summary, checks split leakage, frame
accounting, valid binary targets, and prints per-split acceptance coverage and
positive-frame/positive-video counts. It saves a structured report to
`artifacts/fitness_aqa_qa.json`. Warnings flag low pose coverage and too few
positives; warnings do not rewrite or fabricate the input data.

## B. Full train/validation extraction only

When ready for a real pilot baseline, retain the original source splits and
**exclude test from development**:

```bash
python -m scripts.extract_fitness_aqa_pose \
  --split train-val \
  --max-videos-per-split 0 \
  --sample-fps 2 \
  --min-visibility 0.5
```

The script writes `fitness_aqa_train_pose.csv`,
`fitness_aqa_val_pose.csv` and a new extraction summary. It will not open
the source test clips. ZIP members are temporarily decompressed one at a time,
then deleted. Re-running overwrites the existing train/val CSVs and summary
in the output directory, so back up your pilot output separately if needed.

Review data coverage **before** training:

```bash
python -m scripts.audit_fitness_aqa_pose
```

If the quality gate removes many frames, do not silently lower visibility
to manufacture better results. Report coverage and compare pose extraction
with a clearly recorded gate configuration. More camera positions, occlusion
or poor lighting can explain low acceptance, but individual frames have
not been diagnosed yet.

## C. Controlled two-error Random Forest validation baseline

```bash
python -m scripts.benchmark_fitness_aqa_errors
```

The runner:
- reads **train and val** CSVs only - never test;
- preserves `video_id` and publisher clip splits (not verified subject splits);
- trains separate classifiers for `error_knees_forward` and `error_knees_inward`;
- refuses training when either split has fewer than 20 positive frames,
  5 positive videos or 20 negative frames **for either task**;
- saves models and validation report to `artifacts/fitness_aqa_rf/`;
- reports per-frame macro-F1, error recall/precision, average precision,
  confusion matrix, video-level macro-F1 and per-video metrics.

Default threshold 0.5 is fixed for the initial baseline. The test split
must remain untouched until the validation protocol is frozen. This is a
**camera-based, time-interval error annotation classifier**, not an injury
diagnosis or a guarantee of sound form.

## Important interpretation

`error_knees_forward=0` means the annotators did not mark knees-forward
at that timestamp; it does not certify perfect squat technique. Similarly
for inward knees. There may be other unlabeled movement faults.

Validation precision/F1 on correlated frames should not be interpreted as
independent-person generalization. Even the publisher's disjoint video
keys are **not verified as disjoint people**. Document this limitation in
a portfolio or research report.
