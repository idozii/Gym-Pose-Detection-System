"""Real six-model Web API smoke test with synthetic RF fixtures only.

Checks exact production feature paths and true sklearn prediction
without touching owner data, original publisher TEST, webcams or downloads.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from gym_pose.model_registry import ModelRegistry, KNOWN_TASKS, _contract
from gym_pose.web_api import create_app


def synthetic_landmarks():
    return [
        {
            "x": .11 + .012 * i + (.03 if i % 3 else 0),
            "y": .21 + .013 * (i % 11),
            "z": -.09 + .017 * (i % 7),
            "visibility": .96,
        }
        for i in range(33)
    ]


def test_all_six_web_source_tasks_use_original_rf_contracts(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("POSE_WEB_ORIGINS", "https://demo.example.com")
    rng = np.random.default_rng(11)
    paths = {}
    for exercise, target in KNOWN_TASKS:
        contract = _contract(exercise, target)
        assert contract is not None
        names = contract.feature_names
        x = pd.DataFrame(rng.normal(size=(12, len(names))), columns=names)
        y = np.asarray([0, 1] * 6)
        rf = RandomForestClassifier(n_estimators=3, min_samples_leaf=1,
                                    random_state=42, n_jobs=1)
        rf.fit(x, y)
        bundle = {
            "model": rf,
            "exercise": exercise,
            "target": target,
            "schema_version": contract.schema_version,
            "feature_names": list(names),
            "classes": [0, 1],
            "source_train_only": True,
            "publisher_test_untouched": True,
            "variant": "static",
            contract.threshold_key: .3,
        }
        if exercise == "squat":
            bundle["annotation_semantics"] = (
                "publisher_named_error_interval_NOT_correctness"
            )
        if exercise == "barbell_row":
            bundle["no_direct_lumbar_spine_measurement"] = True
        path = tmp_path / f"{exercise}_{target}.joblib"
        joblib.dump(bundle, path)
        paths[(exercise, target)] = path
    registry = ModelRegistry(paths=paths)
    with TestClient(create_app(registry, trust_local_artifacts=True)) as client:
        health = client.get("/health").json()
        assert len(health["models"]) == 6
        assert set(health["models"].values()) == {"loaded_experimental_only"}
        for exercise in ("squat", "ohp", "barbell_row"):
            response = client.post(
                "/score",
                json={"exercise": exercise, "landmarks": synthetic_landmarks()},
                headers={"Origin": "https://demo.example.com"},
            )
            assert response.status_code == 200
            reports = response.json()["targets"]
            assert len(reports) == 2
            for report in reports.values():
                assert report["status"] == "experimental_source_label_score"
                assert report["reason"] is None
                assert report["threshold"] == .3
                assert 0 <= report["error_probability"] <= 1
                assert isinstance(report["source_label_flag"], bool)
        assert health["raw_camera_frames_processed"] is False
