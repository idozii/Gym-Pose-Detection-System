"""M18 research web API synthetic tests.

Never read publisher TEST clips, decode images, fetch models, or deserialize
the owner's local pickle artifacts. The API's true feature/model path is
smoke-tested separately using synthetic sklearn source-provenance bundles.
"""
from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from gym_pose.model_registry import KNOWN_TASKS
from gym_pose.web_api import create_app


class FakeRegistry:
    def __init__(self, *, loaded=True):
        self.calls = []
        self.loaded = loaded
        self.trust_calls = []
    def load_local(self, *, trust_local_pickle_files=False):
        self.trust_calls.append(trust_local_pickle_files)
        return {
            f"{exercise}/{target}": (
                "loaded_experimental_only" if self.loaded
                else "model_artifact_missing")
            for exercise, target in KNOWN_TASKS
        }
    def score_landmarks(self, exercise, target, landmarks, *, min_visibility=.5):
        self.calls.append((exercise, target, len(landmarks), min_visibility))
        if exercise == "squat" and target == "error_knees_inward":
            return {
                "status": "abstain", "reason": "missing_or_rejected_pose",
                "source_label_flag": None, "error_probability": None,
                "threshold": None,
            }
        return {
            "status": "experimental_source_label_score",
            "source_label_flag": True, "error_probability": .726,
            "threshold": .3, "schema_version": "synthetic_fixture",
        }


def body(exercise="squat"):
    return {
        "exercise": exercise,
        "landmarks": [
            {"x": i / 100., "y": i / 200., "z": -.01 * i,
             "visibility": .99}
            for i in range(33)
        ]
    }


def test_disabled_api_does_not_unpickle_or_score(monkeypatch):
    monkeypatch.setenv("POSE_WEB_ORIGINS", "https://demo.example.com")
    mock = FakeRegistry()
    with TestClient(create_app(mock, trust_local_artifacts=False)) as client:
        h = client.get("/health").json()
        assert h["status"] == "model_artifacts_not_provisioned"
        assert h["pose_estimator_source_parity_validated"] is False
        response = client.post("/score", json=body(),
                               headers={"Origin": "https://demo.example.com"})
        assert response.status_code == 503
        assert mock.calls == [] and mock.trust_calls == []


def test_live_two_original_source_tasks_only_no_echoed_pose(monkeypatch):
    monkeypatch.setenv("POSE_WEB_ORIGINS", "https://demo.example.com")
    mock = FakeRegistry()
    with TestClient(create_app(mock, trust_local_artifacts=True)) as client:
        health = client.get("/health").json()
        assert health["status"] == "research_api_ready"
        assert health["models"]["squat/error_knees_forward"] == "loaded_experimental_only"
        result = client.post(
            "/score", json=body(),
            headers={"Origin": "https://demo.example.com"})
        assert result.status_code == 200
        assert result.headers["Cache-Control"] == "no-store"
        data = result.json()
        assert data["exercise"] == "squat"
        assert set(data["targets"]) == {"error_knees_forward", "error_knees_inward"}
        assert data["targets"]["error_knees_forward"]["error_probability"] == .726
        assert data["targets"]["error_knees_inward"]["status"] == "abstain"
        assert data["camera_frames_retained"] == 0
        assert data["pose_estimator_source_parity_validated"] is False
        assert "landmarks" not in data and "x" not in data
        assert mock.calls == [
            ("squat", "error_knees_forward", 33, .5),
            ("squat", "error_knees_inward", 33, .5),
        ]
        assert mock.trust_calls == [True]


@pytest.mark.parametrize("exercise,expected", [
    ("ohp", {"error_elbows", "error_knees"}),
    ("barbell_row", {"torso_angle", "lumbar"})
])
def test_separate_exercise_target_schema(monkeypatch, exercise, expected):
    monkeypatch.setenv("POSE_WEB_ORIGINS", "http://localhost:5173")
    mock = FakeRegistry()
    with TestClient(create_app(mock, trust_local_artifacts=True)) as client:
        result = client.post("/score", json=body(exercise))
        assert set(result.json()["targets"]) == expected
        assert len(mock.calls) == 2
        assert all(call[0] == exercise for call in mock.calls)


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(exercise="deadlift"),
    lambda x: x.update(landmarks=x["landmarks"][:32]),
    lambda x: x["landmarks"][0].update(visibility=-1),
    lambda x: x["landmarks"][0].update(visibility=1.1),
    lambda x: x["landmarks"][0].update(x=float("inf")),
    lambda x: x["landmarks"][0].update(x=999999),
    lambda x: x["landmarks"][0].update(image="not permitted"),
])
def test_invalid_source_inputs_never_reach_registry(monkeypatch, mutation):
    monkeypatch.setenv("POSE_WEB_ORIGINS", "http://localhost:5173")
    mock = FakeRegistry()
    with TestClient(create_app(mock, trust_local_artifacts=True)) as client:
        item = body()
        mutation(item)
        result = client.post("/score", json=item)
        assert result.status_code == 422
        assert mock.calls == []


def test_other_origin_rejected_and_no_external_wildcard(monkeypatch):
    monkeypatch.setenv("POSE_WEB_ORIGINS", "https://demo.example.com")
    mock = FakeRegistry()
    with TestClient(create_app(mock, trust_local_artifacts=True)) as client:
        result = client.post("/score", json=body(),
                             headers={"Origin":"https://malicious.example"})
        assert result.status_code == 403
        assert mock.calls == []
        allowed = client.post("/score", json=body(),
                              headers={"Origin":"https://demo.example.com"})
        assert allowed.headers.get("access-control-allow-origin") == "https://demo.example.com"


def test_large_request_is_rejected_before_model(monkeypatch):
    monkeypatch.setenv("POSE_WEB_ORIGINS", "http://localhost:5173")
    mock = FakeRegistry()
    with TestClient(create_app(mock, trust_local_artifacts=True)) as client:
        result = client.post(
            "/score", content="x"*17000,
            headers={"Content-Type":"application/json"})
        assert result.status_code == 413
        assert mock.calls == []


def test_registry_error_abstains_without_logging_error_details(monkeypatch):
    class Broken(FakeRegistry):
        def score_landmarks(self, exercise, target, landmarks, **kwargs):
            raise ValueError("SENSITIVE INTERNAL MODEL FILE PATH")
    monkeypatch.setenv("POSE_WEB_ORIGINS", "http://localhost:5173")
    mock = Broken()
    with TestClient(create_app(mock, trust_local_artifacts=True)) as client:
        result = client.post("/score", json=body()).json()
        assert all(r["status"] == "abstain" for r in result["targets"].values())
        assert "SENSITIVE" not in str(result)
        assert all(r["reason"] == "model_inference_error" for r in result["targets"].values())


def test_no_open_cors_wildcard_on_production(monkeypatch):
    monkeypatch.setenv("POSE_WEB_ORIGINS", "*")
    with pytest.raises(ValueError, match="HTTPS"):
        create_app(FakeRegistry(), trust_local_artifacts=False)
