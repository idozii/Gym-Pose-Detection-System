# Fitness-AQA Validation Diagnostics: before upgrading the model

## Baseline results (owner's real Fitness-AQA train/val extraction)

Two independent Random Forest models were trained on 4,309 retained training
frames and evaluated on 984 retained validation frames (not independent persons).
Default decision threshold in both models: 0.5.

| Error annotation | Validation positive frames | Frame macro-F1 | Precision (error) | Recall (error) |
| --- | ---: | ---: | ---: | ---: |
| Knee forward | 214 | 0.5841 | 0.3780 | 0.2897 |
| Knee inward | 40 | 0.4891 | 0.0000 | 0.0000 |

The knee-inward model detected **zero of forty** annotated positive frames.
The knee-forward model detected **62 of 214** positive frames but raised
102 false alarms on negative-annotation frames. These numbers should not be
reported as validated exercise-form quality or medical safety measures.

## Why diagnose probabilities before training a larger model?

An RF can have useful **ranking quality** despite probabilities that mostly
fall below a fixed 0.5 cutoff. Conversely, if ranking quality is poor,
lowering threshold increases false alarms without solving feature/model issues.

Run directly on already-extracted validation CSV and **trusted locally trained**
model files. No video decoding, no re-training and no test data:

```bash
git fetch origin
git switch feat/fitness-aqa-threshold-diagnostics
python -m scripts.diagnose_fitness_aqa_thresholds
```

Outputs (Git-ignored):

```text
artifacts/fitness_aqa_diagnostics/
  threshold_diagnostics.json
  threshold_grid.csv
```

For each target, the JSON includes:
- Average precision vs the positive-class prevalence baseline.
- Positive/negative probability 10th, median and 90th percentiles.
- Fixed 0.5 threshold confusion matrix entries and precision/recall/macro-F1.
- Candidate selected by *validation* frame macro-F1 over a predeclared grid
  0.05, 0.10, ... 0.95. This is **exploratory**, not an unbiased score.
- Clip-level "any error in at least one accepted frame" diagnostics, separate
  from the old average of per-clip frame F1. This too misses errors in rejected
  frames and does not measure continuous event detection.

Decision guidance:
1. If average precision substantially exceeds positive prevalence and error
   probability quantiles differ, a lower threshold may be worth evaluating.
2. If average precision is near prevalence or distributions overlap, improve
   pose quality, camera/view robustness and temporal features first.
3. Compare threshold impact on false positive rate **and** false negatives;
   never choose on recall alone.
4. Avoid cherry-picking a threshold to maximize the same validation score.
   Freeze the operating point and evaluation protocol before a single final
   test assessment. Publisher train/val/test video IDs are not verified
   subject-disjoint.

Do not paste a downloaded/untrusted `.joblib` into the tool: loading it with
joblib/pickle may execute arbitrary code. Use the model artifacts you trained
locally with `scripts.benchmark_fitness_aqa_errors`.
