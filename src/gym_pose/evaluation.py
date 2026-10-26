"""Frozen subject-disjoint holdout with honest generalization reporting."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from .features import FEATURE_NAMES, SCHEMA_VERSION
from .training import LABELS, load_training_data


def _check_provenance(df: pd.DataFrame) -> None:
    required = ["subject_id", "session_id", "camera_view", "sample_id", "image_sha256"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required provenance missing: {missing}. Use manifest extraction.")
    if df[required].isna().any().any() or (df[required].astype(str).apply(lambda x: x.str.strip()) == "").any().any():
        raise ValueError("Incomplete provenance")
    if df.sample_id.duplicated().any() or df.image_sha256.duplicated().any():
        raise ValueError("Duplicate sample ID or image content")
    if (df.groupby("session_id").subject_id.nunique() > 1).any():
        raise ValueError("Session IDs shared between subjects")


def make_split(df: pd.DataFrame, *, test_size: float = .2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    _check_provenance(df)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    if df.subject_id.nunique() < 4:
        raise ValueError("At least four independent subjects required")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(splitter.split(df, df.label, df.subject_id))
    if set(df.iloc[tr].label) != set(LABELS) or set(df.iloc[te].label) != set(LABELS):
        raise ValueError("Both labels required in both subject-disjoint groups")
    assert not set(df.iloc[tr].subject_id).intersection(df.iloc[te].subject_id)
    return tr, te


def _metrics(y_true, y_pred) -> dict:
    if len(y_true) == 0:
        return {"n": 0}
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=list(LABELS), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
    }


def benchmark(features_csv: str | Path, output_dir: str | Path, *, seed: int = 42,
              test_size: float = .2) -> dict:
    df = load_training_data(features_csv)
    _check_provenance(df)
    tr, te = make_split(df, test_size=test_size, seed=seed)
    X = df.loc[:, list(FEATURE_NAMES)].astype(float)
    y = df.label
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed,
                                   n_jobs=-1, class_weight="balanced")
    model.fit(X.iloc[tr], y.iloc[tr])
    pred = model.predict(X.iloc[te])
    test = df.iloc[te].copy()
    test["prediction"] = pred
    overall = _metrics(test.label, test.prediction)
    per_view = {str(view): _metrics(part.label, part.prediction)
                for view, part in test.groupby("camera_view", sort=True)}
    per_subject = {str(subject): _metrics(part.label, part.prediction)
                   for subject, part in test.groupby("subject_id", sort=True)}
    manifest_digest = hashlib.sha256(
        df[["sample_id", "image_sha256", "subject_id", "label"]]
        .sort_values("sample_id").to_csv(index=False).encode()
    ).hexdigest()
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "measured_on_input_data",
        "split_method": "single_holdout_subject_disjoint",
        "seed": seed, "test_size": test_size, "dataset_fingerprint": manifest_digest,
        "train_n": len(tr), "test_n": len(te),
        "train_subjects": sorted(df.iloc[tr].subject_id.astype(str).unique().tolist()),
        "test_subjects": sorted(test.subject_id.astype(str).unique().tolist()),
        "label_order": list(LABELS),
        "overall": overall, "by_camera_view": per_view, "by_subject": per_subject,
        "warnings": [
            "One seeded holdout is not a confidence interval or cross-dataset validation",
            "No model selection/tuning performed; results apply only to these images/subjects",
            "Camera views with few observations are unstable",
            "Samples filtered by pose visibility may change evaluation coverage",
        ]
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    test.loc[:, ["sample_id", "subject_id", "session_id", "camera_view", "label", "prediction"]].to_csv(target / "test_predictions.csv", index=False)
    joblib.dump({
        "model": model, "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES), "classes": list(model.classes_),
        "exercise": "squat", "feature_count": len(FEATURE_NAMES),
    }, target / "squat_rf_v1.joblib")
    return result
