# Milestone 2 — Dataset provenance and subject-independent benchmark

The code and tests implement a **workflow**, not a claim that the current squat model
generalizes. No real user data or measured real-world scores ship with this PR.

## 1. Define and verify your data
Images remain local and are ignored by Git. Use consented, genuinely annotated images.
```bash
python -m pip install -e '.[camera,dev]'
gym-pose manifest-template
```
Edit `data/manifest.csv` and add one row per actual image.

Required fields:
- `sample_id`: stable unique pseudonymous identifier for each image
- `relative_path`: image path **relative to** `--image-root`, e.g. `squat/correct/example1.jpg`
- `exercise`: `squat` (only supported exercise)
- `label`: `correct` or `incorrect`; these are **annotation judgments**, not injury diagnoses
- `subject_id`: genuine stable pseudonymous person identifier; must not be assigned randomly per image
- `session_id`: unique capture session identifier, belonging to exactly one subject
- `camera_view`: `front`, `side`, `diagonal`, or `other`

Example **format only** (do NOT treat this as data):
```csv
sample_id,relative_path,exercise,label,subject_id,session_id,camera_view
s0001,squat/correct/001.jpg,squat,correct,subject_a,session_a,side
s0002,squat/incorrect/002.jpg,squat,incorrect,subject_b,session_b,front
```
Files must actually exist under data/raw for validation.

```bash
gym-pose validate-manifest --manifest data/manifest.csv --image-root data/raw
gym-pose extract-manifest --manifest data/manifest.csv --image-root data/raw
```
The validator checks required fields, allowed labels/views, unique sample IDs,
duplicate paths, SHA-256 identical image bytes, no path traversal and session-to-subject consistency.
The extractor checks image digests again and writes 103-feature observations plus provenance.
It reports images skipped for pose quality: review skipped rates by group and label.

## 2. Run the holdout benchmark
```bash
gym-pose benchmark \
  --features data/features/squat_provenance_v1.csv \
  --output-dir artifacts/benchmark \
  --seed 42
```
Outputs:
- `benchmark.json`: macro-F1, accuracy, confusion matrix, per-view/per-subject reports, fingerprint, split metadata and limitations
- `test_predictions.csv`: traceable held-out predictions
- `squat_rf_v1.joblib`: the benchmark model, compatible with the Milestone 1 webcam loader

A subject-disjoint holdout ensures an individual does not appear on both sides.
It does not guarantee camera-domain generalization or provide a meaningful score
with few subjects. At least four unique subjects and both labels in each split
are required. One holdout is not nested CV or confidence intervals.

## 3. Data policy / limitations
- No videos, photos or biological identifiers should be committed or shared without consent.
- Use subject IDs that are consistent across recordings, and preserve original dataset licensing.
- Never fabricate subject/session/camera labels for a legacy CSV; recollect and reannotate.
- Exact-file duplicate checking does **not** catch near-duplicate adjacent video frames.
- Image-level classification can reward redundant frames; expand to rep-level
  labels and independent videos in later experiments.
- Training/test examples filtered due to bad pose change coverage: report this alongside performance.
- To avoid choosing a lucky holdout, later add a frozen test set and nested/grouped CV.

## 4. Development tests
```bash
python -m pytest -q
```
Tests generate **synthetic arrays and dummy image bytes** for software checks
only. They are not a benchmark or a real-world model-performance estimate.
