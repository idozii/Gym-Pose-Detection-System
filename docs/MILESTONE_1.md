# Milestone 1 - Modular squat baseline

## Scope
This is a **refactor and reliability milestone**, not an accuracy improvement.
The original notebook and existing binary model are preserved in the default branch.
The new CLI requires a newly trained, explicitly versioned model.

## Setup (Python 3.10, from repository root)
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[camera,dev]'
pytest -q
```

## Capture, train and evaluate
```bash
gym-pose capture --label correct --count 100
gym-pose capture --label incorrect --count 100
gym-pose extract
gym-pose train
gym-pose webcam
```
Run in a desktop session with a working webcam. Inference is local and is not medical or personalized coaching.

**Important:** the default training split is frame-random and explicitly marked
exploratory, because the current notebook does not record subject/session IDs.
Do not interpret its accuracy as independent-person generalization.

To enable a grouped split, add a `subject_id` or `session_id` column
to the feature CSV with trustworthy identifiers and invoke
`gym-pose train --group-column subject_id`. Grouped splitting requires
both classes in train/test. Use video/session-level provenance, not invented IDs.

## Compatibility and safety
- 103-feature numeric schema matches the notebook mathematics but now has named columns.
- Model metadata carries feature order, classes and schema version.
- Legacy, unversioned joblib models must be retrained, not blindly reused.
- Low visibility, NaN, missing joints and degenerate shoulders abstain.
- Probability smoothing uses estimator class order and resets on lost pose.
- The model is a demonstration, not a validated injury-prevention system.
- Do not load downloaded/untrusted joblib models; pickle deserialization can execute code.
- Data collected by webcam stays local by default, but consent is required for other people.

## Known limitations
One view, squat only, 2 classes, no temporal rep analysis or subject-disjoint dataset.
The notebook remains an historical baseline; it has not been modified to use the new package.
Video capture/extraction still needs MediaPipe's legacy solutions API (pinned 0.10.14).
