"""Explicitly labeled baseline training and deterministic evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .features import FEATURE_NAMES, SCHEMA_VERSION

LABELS = ("correct", "incorrect")


def load_training_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Migrate the notebook's f0..f102 CSVs without changing feature semantics.
    legacy_names = [f"f{i}" for i in range(len(FEATURE_NAMES))]
    if set(legacy_names).issubset(df.columns) and not set(FEATURE_NAMES).issubset(df.columns):
        df = df.rename(columns=dict(zip(legacy_names, FEATURE_NAMES)))
    missing = set(FEATURE_NAMES).difference(df.columns)
    if missing or "label" not in df:
        raise ValueError(f"Missing features: {sorted(missing)}; 'label' is required")
    if df.empty or set(df.label.dropna().unique()) != set(LABELS) or df.label.isna().any():
        raise ValueError("Nonempty data with both 'correct' and 'incorrect' labels required")
    df = df.copy()
    df.loc[:, list(FEATURE_NAMES)] = df.loc[:, list(FEATURE_NAMES)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite features are not supported")
    return df


def train(csv_path: str | Path, model_path: str | Path, *, group_column: str | None = None,
          test_size: float = 0.2, seed: int = 42) -> dict:
    df = load_training_data(csv_path)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    features = df.loc[:, list(FEATURE_NAMES)].astype("float64")
    targets = df["label"]
    if group_column:
        if group_column not in ("subject_id", "session_id") or group_column not in df.columns:
            raise ValueError("group_column must be an existing subject_id or session_id")
        if df[group_column].isna().any() or df[group_column].nunique() < 2:
            raise ValueError("At least two nonmissing groups are required")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(features, targets, groups=df[group_column]))
        split_name = f"group_disjoint:{group_column}"
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=test_size, random_state=seed, stratify=targets
        )
        split_name = "random_frames_EXPLORATORY_ONLY"
    if set(targets.iloc[train_idx]) != set(LABELS) or set(targets.iloc[test_idx]) != set(LABELS):
        raise ValueError("Both classes must exist in train and test; collect more groups/data")
    classifier = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1,
        class_weight="balanced"
    )
    classifier.fit(features.iloc[train_idx], targets.iloc[train_idx])
    predictions = classifier.predict(features.iloc[test_idx])
    report = {
        "schema_version": SCHEMA_VERSION,
        "split": split_name,
        "research_ready": bool(group_column),
        "warning": "Frame split may leak subject/session appearance; not a generalization estimate"
        if not group_column else "Grouped split is a first step; separate sessions/views may still leak",
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "classification_report": classification_report(
            targets.iloc[test_idx], predictions, labels=list(LABELS), output_dict=True,
            zero_division=0
        ),
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(
            targets.iloc[test_idx], predictions, labels=list(LABELS)
        ).tolist(),
    }
    bundle = {
        "model": classifier, "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES), "classes": list(classifier.classes_),
        "exercise": "squat", "feature_count": len(FEATURE_NAMES)
    }
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def load_model(path: str | Path) -> dict:
    """Only load artifacts you have trained and trust: joblib uses pickle."""
    bundle = joblib.load(path)
    if (not isinstance(bundle, dict) or bundle.get("schema_version") != SCHEMA_VERSION
        or bundle.get("feature_names") != list(FEATURE_NAMES)
        or bundle.get("feature_count") != len(FEATURE_NAMES)
        or bundle.get("classes") != list(getattr(bundle.get("model"), "classes_", []))):
        raise ValueError("Incompatible or legacy model; retrain with 'gym-pose train'")
    return bundle
