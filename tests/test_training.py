from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from gym_pose.features import FEATURE_NAMES
from gym_pose.training import load_model, load_training_data, train


def synthetic_csv(tmp_path, *, with_groups=False):
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame(rng.normal(size=(n, 103)), columns=list(FEATURE_NAMES))
    df["label"] = ["correct", "incorrect"] * 40
    if with_groups:
        df["subject_id"] = [f"s{i // 8}" for i in range(n)]
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)
    return path


def test_legacy_columns_migrate(tmp_path):
    path = synthetic_csv(tmp_path)
    df = pd.read_csv(path)
    df.rename(columns=dict(zip(FEATURE_NAMES, [f"f{i}" for i in range(103)]))).to_csv(path, index=False)
    assert list(load_training_data(path).loc[:, list(FEATURE_NAMES)].columns) == list(FEATURE_NAMES)


def test_train_save_and_load(tmp_path):
    path = synthetic_csv(tmp_path)
    target = tmp_path / "model.joblib"
    report = train(path, target)
    assert report["research_ready"] is False
    assert report["split"] == "random_frames_EXPLORATORY_ONLY"
    assert load_model(target)["classes"] == ["correct", "incorrect"]
    assert target.with_suffix(".metrics.json").exists()


def test_grouped_split(tmp_path):
    path = synthetic_csv(tmp_path, with_groups=True)
    result = train(path, tmp_path / "group.joblib", group_column="subject_id")
    assert result["split"] == "group_disjoint:subject_id"


def test_unversioned_bundle_is_rejected(tmp_path):
    target = tmp_path / "old.joblib"
    joblib.dump({"model": object()}, target)
    with pytest.raises(ValueError):
        load_model(target)


def test_reject_missing_labels(tmp_path):
    path = synthetic_csv(tmp_path)
    df = pd.read_csv(path)
    df["label"] = "correct"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_training_data(path)
