import hashlib

import numpy as np
import pandas as pd
import pytest
from gym_pose.evaluation import make_split, benchmark
from gym_pose.features import FEATURE_NAMES


def sample_features(tmp_path, *, subjects=10):
    rng = np.random.default_rng(7)
    data = pd.DataFrame(rng.standard_normal((subjects * 8, 103)), columns=FEATURE_NAMES)
    n = len(data)
    data["sample_id"] = [f"image{i}" for i in range(n)]
    data["image_sha256"] = [hashlib.sha256(f"image{i}".encode()).hexdigest() for i in range(n)]
    data["subject_id"] = [f"person{i // 8}" for i in range(n)]
    data["session_id"] = [f"session{i // 8}" for i in range(n)]
    data["camera_view"] = ["front", "side"] * (n // 2)
    data["label"] = ["correct", "incorrect"] * (n // 2)
    path = tmp_path / "features.csv"
    data.to_csv(path, index=False)
    return path


def test_subjects_never_overlap(tmp_path):
    path = sample_features(tmp_path)
    df = pd.read_csv(path)
    train, test = make_split(df)
    assert set(df.iloc[train].subject_id).isdisjoint(set(df.iloc[test].subject_id))


def test_benchmark_generates_reproducible_report(tmp_path):
    path = sample_features(tmp_path)
    result = benchmark(path, tmp_path / "report")
    repeat = benchmark(path, tmp_path / "repeat")
    assert result["dataset_fingerprint"] == repeat["dataset_fingerprint"]
    assert result["test_subjects"] == repeat["test_subjects"]
    assert result["overall"]["macro_f1"] == repeat["overall"]["macro_f1"]
    assert set(result["by_camera_view"]) == {"front", "side"}
    assert (tmp_path / "report" / "benchmark.json").exists()
    assert (tmp_path / "report" / "test_predictions.csv").exists()


def test_missing_provenance_is_rejected(tmp_path):
    path = sample_features(tmp_path)
    df = pd.read_csv(path).drop(columns=["subject_id"])
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Required provenance"):
        benchmark(path, tmp_path / "out")


def test_duplicate_image_hash_rejected(tmp_path):
    path = sample_features(tmp_path)
    df = pd.read_csv(path)
    df.loc[1, "image_sha256"] = df.loc[0, "image_sha256"]
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark(path, tmp_path / "out")
