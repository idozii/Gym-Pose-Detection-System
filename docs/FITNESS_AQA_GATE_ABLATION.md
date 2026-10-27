# Fitness-AQA: controlled visibility gate ablation (0.50 versus 0.35)

## Evidence from the owner's stratified 90-clip quality audit

On 635 eligible sampled frames (45 train + 45 validation clips), the production
0.50 visibility gate accepted 374 (58.9%). Of 261 rejected frames,
196 (75.1%) failed critical joint visibility, and 65 (24.9%) had no
detected pose. No other rejection type occurred. Low visibility most often
implicated knees and ankles; an individual frame can fail multiple joints.

The gate sensitivity counts for the *same sampled frames* were:
- 0.65: 295/635 = 46.5%
- 0.50: 374/635 = 58.9%
- 0.35: 454/635 = 71.5%
- 0.25: 496/635 = 78.1%

This audit deliberately oversampled error-annotated videos; it is **not** an
estimate of overall Fitness-AQA coverage. Inward-positive validation frames
had 13/30=43.3% acceptance versus 178/286=62.2% among
inward-negative frames **in this stratified audit only**.

This is sufficient motivation to test 0.35, **not evidence** it will
improve model quality.

## Exact next experiment - do not overwrite the original 0.50 dataset

From the repository root:
```bash
git fetch origin
git switch --track origin/feat/fitness-aqa-visibility-gate-ablation
python -m pip install -e '.[camera,dev]'

# Re-extract source TRAIN + VALIDATION at 0.35 to a DIFFERENT directory.
# This deliberately excludes test and does not change current 0.50 CSVs.
python -m scripts.extract_fitness_aqa_pose \
  --split train-val --max-videos-per-split 0 \
  --sample-fps 2 --boundary-margin 0.1 \
  --min-visibility 0.35 \
  --output-dir data/features/fitness_aqa_gate035

# Compare paired 0.50 and 0.35 data with identical source video fit/cal groups.
python -m scripts.compare_fitness_aqa_visibility
```

Reports are Git-ignored and small:
`artifacts/fitness_aqa_gate_ablation/comparison.json`.

## Fairness controls and limitations

- Load your existing `data/features/fitness_aqa` (0.50) and new
  `data/features/fitness_aqa_gate035` (0.35).
- Assert matching original source sampling FPS, annotation-boundary
  margin, frame counts, and train/validation clips processed.
- Check intersection of `sample_id` across the two extractions;
  record original-only/relaxed-only frames and numerical feature
  differences for common frames. Re-running MediaPipe separately may
  introduce run-to-run variation, so the comparison is not perfectly controlled.
- Train the 119-feature **causal temporal** RF using each gate's
  source train data, but select thresholds on the **same** train-video
  group calibration IDs. Refit each RF on its entire own train set.
- On validation, report error-specific AP, macro-F1, precision, recall,
  and false positive rate on both (a) **all** accepted frames at
  each gate and (b) the **identical shared accepted frame IDs**.
- Raw metrics on the two *different* full accepted validation sets
  must NOT be directly treated as a causal model improvement.
  The common-frame comparison is more interpretable, but the training
  populations and temporal histories still differ.
- Examine positive-frame counts and coverage changes alongside scores.
  No test CSVs, videos or frame-level test labels are read.

## Decision rubric

**Prefer 0.35 only if** it adds useful error examples and keeps or
improves AP/recall and false-alert rate on common validation observations,
without unacceptable degradation. The new frames may have less reliable
landmarks; their contribution to a live system needs visual/pose checks.

If apparent gains occur only because the 0.35 gate changes which frames
get evaluated, do not claim a better classifier.

After exploratory validation, freeze the preprocessing, training, threshold
and evaluation protocol before running the publisher **test once**. Publisher
video-ID disjointness does not establish subject independence. Class labels
refer only to specifically annotated intervals; do not claim injury prevention.
