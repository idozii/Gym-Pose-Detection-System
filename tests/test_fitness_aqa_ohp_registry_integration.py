"""M15B: source OHP 30-column adapter integrated into research registry.

Synthetic fitted Random Forests only. Never decodes original videos, loads
untrusted artifacts, or touches source TRAIN/VAL/TEST dataset files.
"""
from __future__ import annotations

from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from gym_pose.model_registry import ModelRegistry, _contract
from gym_pose.ohp_task_features import (
    FEATURES_BY_TASK, JOINTS_BY_TASK, SCHEMA_BY_TASK, TASKS,
    pose_to_task_features,
)


def landmarks(*, low=()):
    return [
        SimpleNamespace(x=.02 * i, y=.03 * i, z=.001 * i,
                        visibility=.1 if i in low else .99)
        for i in range(33)
    ]


def fake_ohp_bundle(tmp_path, task, **updates):
    features = list(FEATURES_BY_TASK[task])
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(40, len(features))), columns=features)
    y = [0] * 20 + [1] * 20
    fitted = RandomForestClassifier(n_estimators=8, random_state=42, n_jobs=1)
    fitted.fit(x, y)
    bundle = {
        "model": fitted, "exercise": "ohp", "target": task,
        "variant": "static",
        "schema_version": SCHEMA_BY_TASK[task],
        "feature_names": features,
        "classes": [0, 1], "threshold_selected_on_train_groups_only": .25,
        "source_train_only": True, "publisher_test_untouched": True,
    }
    bundle.update(updates)
    path = tmp_path / f"{task}_static.joblib"
    joblib.dump(bundle, path)
    return path


def test_original_m13c_schema_is_unmodified_and_task_specific():
    assert TASKS == ("error_elbows", "error_knees")
    assert SCHEMA_BY_TASK == {
        "error_elbows": "ohp-elbows-upper8-30-v1",
        "error_knees": "ohp-knees-lower8-30-v1",
    }
    for task in TASKS:
        assert len(FEATURES_BY_TASK[task]) == 30
        assert len(set(FEATURES_BY_TASK[task])) == 30
        result = pose_to_task_features(landmarks(), task=task)
        assert result is not None
        assert tuple(result) == FEATURES_BY_TASK[task]
        assert np.isfinite(list(result.values())).all()
        contract = _contract("ohp", task)
        assert contract is not None
        assert contract.feature_names == FEATURES_BY_TASK[task]
        assert contract.schema_version == SCHEMA_BY_TASK[task]

    low_legs = landmarks(low=(25, 26, 27, 28))
    assert pose_to_task_features(low_legs, task="error_elbows") is not None
    assert pose_to_task_features(low_legs, task="error_knees") is None
    low_arms = landmarks(low=(13, 14, 15, 16))
    assert pose_to_task_features(low_arms, task="error_elbows") is None
    assert pose_to_task_features(low_arms, task="error_knees") is not None


def test_two_OHP_models_load_and_score_with_local_trust_only(tmp_path):
    paths = {("ohp", t): fake_ohp_bundle(tmp_path, t) for t in TASKS}
    registry = ModelRegistry(paths=paths)
    with pytest.raises(ValueError, match="explicitly opt"):
        registry.load_local()
    assert registry._models == {}
    states = registry.load_local(trust_local_pickle_files=True)
    assert all(states[f"ohp/{t}"] == "loaded_experimental_only" for t in TASKS)
    assert states["barbell_row/lumbar"] == "model_artifact_missing"
    assert states["squat/error_knees_forward"] == "not_trained_as_source_error_detector"
    for task in TASKS:
        p = registry.score_landmarks("ohp", task, landmarks())
        assert p["status"] == "experimental_source_label_score"
        assert p["schema_version"] == SCHEMA_BY_TASK[task]
        assert p["feature_count"] == 30
        assert p["threshold"] == .25
        assert 0. <= p["error_probability"] <= 1.
        assert isinstance(p["source_label_flag"], bool)
        assert p["experimental_only"]
        assert registry.score_landmarks(
            "ohp", task, landmarks(low=(JOINTS_BY_TASK[task][0],)))["status"] == "abstain"


@pytest.mark.parametrize("updates", [
    {"schema_version": "ohp-legacy-111-v1"},
    {"variant": "causal_temporal"},
    {"feature_names": list(reversed(FEATURES_BY_TASK["error_elbows"]))},
    {"threshold_selected_on_train_groups_only": None},
    {"threshold_selected_on_train_groups_only": float("nan")},
    {"publisher_test_untouched": False},
    {"source_train_only": False},
    {"target": "error_knees"},
])
def test_model_contract_rejects_wrong_variant_or_provenance(tmp_path, updates):
    path = fake_ohp_bundle(tmp_path, "error_elbows", **updates)
    reg = ModelRegistry(paths={("ohp", "error_elbows"): path})
    with pytest.raises(ValueError):
        reg.load_local(trust_local_pickle_files=True)
    assert reg.score_landmarks("ohp", "error_elbows", landmarks())["status"] == "abstain"


def test_missing_artifacts_do_not_fabricate_scores(tmp_path):
    registry = ModelRegistry(paths={
        ("ohp", task): tmp_path / f"absent_{task}.joblib"
        for task in TASKS
    })
    states = registry.load_local(trust_local_pickle_files=True)
    assert all(states[f"ohp/{task}"] == "model_artifact_missing" for task in TASKS)
    for task in TASKS:
        assert registry.score_landmarks("ohp", task, landmarks())["source_label_flag"] is None
