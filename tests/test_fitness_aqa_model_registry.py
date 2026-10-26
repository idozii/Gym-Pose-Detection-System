"""No original fitness dataset, source TEST, or external model artifacts loaded."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from gym_pose import model_registry as reg
from gym_pose.barbell_row_torso4_features import (
    FEATURE_NAMES, SCHEMA_VERSION, pose_to_torso4_features,
)


def landmarks(*, low=()):
    return [
        SimpleNamespace(x=.03*i, y=.015*(i + (i % 3)), z=.001*i,
                        visibility=(.2 if i in low else .98))
        for i in range(33)
    ]


def create_local_artifact(tmp_path, target="lumbar", **overrides):
    rng = np.random.default_rng(14)
    arr = rng.normal(size=(64, len(FEATURE_NAMES)))
    y = np.asarray([0]*32+[1]*32)
    fit = RandomForestClassifier(
        n_estimators=8, random_state=42, n_jobs=1).fit(
        pd.DataFrame(arr, columns=list(FEATURE_NAMES)), y)
    bundle = {
        "model": fit,
        "exercise": "barbell_row", "target": target,
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "source_train_only": True,
        "publisher_test_untouched": True,
        "threshold_train_prefix_calibration": .3,
        "no_direct_lumbar_spine_measurement": True,
    }
    bundle.update(overrides)
    file = tmp_path / "barbell_row_fake_trusted.joblib"
    joblib.dump(bundle, file)
    return file


def test_missing_models_and_squat_source_error_slots_fail_closed():
    registry = reg.ModelRegistry()
    assert len(reg.KNOWN_TASKS) == 6
    for exercise, task in reg.KNOWN_TASKS:
        out = registry.score_features(exercise, task, {})
        assert out["status"] == "abstain"
        assert out["source_label_flag"] is None
        assert out["error_probability"] is None
    assert reg._contract("squat","error_knees_forward") is None
    with pytest.raises(ValueError, match="Unknown"):
        registry.score_features("bench_press","not_a_task",{})


def test_pickle_deserialization_requires_explicit_user_optin(tmp_path):
    file = create_local_artifact(tmp_path)
    registry = reg.ModelRegistry(paths={("barbell_row","lumbar"):file})
    with pytest.raises(ValueError, match="explicitly opt"):
        registry.load_local()
    assert registry._models == {}
    assert registry.score_landmarks("barbell_row","lumbar",landmarks())["status"] == "abstain"


def test_row_model_loads_and_predicts_source_labels_only(tmp_path):
    file = create_local_artifact(tmp_path)
    registry = reg.ModelRegistry(paths={("barbell_row","lumbar"):file})
    states = registry.load_local(trust_local_pickle_files=True)
    assert states["barbell_row/lumbar"] == "loaded_experimental_only"
    assert states["squat/error_knees_forward"] == "not_trained_as_source_error_detector"
    p = registry.score_landmarks("barbell_row","lumbar",landmarks())
    assert p["status"] == "experimental_source_label_score"
    assert p["schema_version"] == SCHEMA_VERSION
    assert p["feature_count"] == 20
    assert p["threshold"] == .3
    assert 0 <= p["error_probability"] <= 1
    assert isinstance(p["source_label_flag"], bool)
    assert p["experimental_only"]
    assert "NOT verified" in p["warning"]
    no_pose = registry.score_landmarks("barbell_row","lumbar",
                                       landmarks(low=(11,)))
    assert no_pose["status"] == "abstain"
    assert no_pose["source_label_flag"] is None


def test_bad_features_or_unknown_task_cannot_trigger_source_label_flag(tmp_path):
    file = create_local_artifact(tmp_path)
    registry = reg.ModelRegistry(paths={("barbell_row","lumbar"):file})
    registry.load_local(trust_local_pickle_files=True)
    pose = pose_to_torso4_features(landmarks())
    assert pose is not None
    with_missing = dict(pose)
    with_missing.pop(next(iter(with_missing)))
    bad = dict(pose)
    bad[next(iter(bad))] = float("nan")
    for candidate in (None, with_missing, bad, {"unknown":1}, [1,2,3]):
        result = registry.score_features("barbell_row","lumbar",candidate)
        assert result["status"] == "abstain"
        assert result["source_label_flag"] is None
    shuffled = dict(reversed(list(pose.items())))
    assert registry.score_features("barbell_row","lumbar",shuffled)[
        "status"] == "experimental_source_label_score"
    with pytest.raises(ValueError, match="Unknown"):
        reg._contract("unknown","error_x")


@pytest.mark.parametrize("metadata_override", [
    {"target":"torso_angle"},
    {"schema_version":"barbell-row-image-32-v1"},
    {"feature_names":list(reversed(FEATURE_NAMES))},
    {"threshold_train_prefix_calibration": float("nan")},
    {"threshold_train_prefix_calibration": 1.},
    {"threshold_train_prefix_calibration": False},
    {"source_train_only":False},
    {"publisher_test_untouched":False},
    {"no_direct_lumbar_spine_measurement":False},
    {"model":None},
    {"classes":[1,0]},
])
def test_fail_closed_on_corrupt_or_mislabeled_artifacts(tmp_path,metadata_override):
    file = create_local_artifact(tmp_path, **metadata_override)
    registry = reg.ModelRegistry(paths={("barbell_row","lumbar"):file})
    with pytest.raises(ValueError):
        registry.load_local(trust_local_pickle_files=True)
    assert registry.score_features("barbell_row","lumbar",{})["status"] == "abstain"


def test_registry_paths_refuse_masqueraded_squat_models(tmp_path):
    with pytest.raises(ValueError, match="No trained artifact contract"):
        reg.ModelRegistry(paths={("squat","error_knees_forward"):tmp_path/"model.joblib"})
