"""M18 owner deployment readiness preflight tests: never contact the internet."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from scripts.check_web_inference_deployment import (
    EXPECTED, canonical_api_url, check_health_payload, main,
)


def complete_health():
    return {
        "status": "research_api_ready",
        "research_only": True,
        "pose_estimator_source_parity_validated": False,
        "ready_for_six_detector_demo": True,
        "loaded_model_count": 6,
        "expected_model_count": 6,
        "models": {name: "loaded_experimental_only" for name in EXPECTED},
    }


def test_strict_readiness_accepts_only_six_expected_source_models():
    assert check_health_payload(complete_health())["ready"] is True
    partial = complete_health()
    partial["models"]["ohp/error_knees"] = "model_artifact_missing"
    partial["loaded_model_count"] = 5
    partial["ready_for_six_detector_demo"] = False
    result = check_health_payload(partial)
    assert result["ready"] is False
    assert result["unavailable"] == ["ohp/error_knees"]
    added = complete_health()
    added["models"]["fake/fitness_safety"] = "loaded_experimental_only"
    assert check_health_payload(added)["ready"] is False


@pytest.mark.parametrize("bad", [
    "http://public.example.com", "https://example.com/path",
    "https://name:secret@example.com", "https://example.com?x=1",
    "https://example.com#part", "https://",
])
def test_never_send_requests_to_invalid_or_untrusted_api_origins(bad):
    with pytest.raises(ValueError):
        canonical_api_url(bad)


def test_localhost_development_is_allowed():
    assert canonical_api_url("http://localhost:8000/") == "http://localhost:8000"


def test_cli_probes_health_and_ready_without_sending_pose(capsys):
    endpoints = []

    def mock_read(url, *, timeout):
        endpoints.append(url)
        return complete_health() if url.endswith("/health") else {
            "status": "ready", "loaded_model_count": 6
        }

    with patch("scripts.check_web_inference_deployment.read_json", side_effect=mock_read):
        assert main(["--api-url", "https://api.example.com"]) == 0
    assert endpoints == ["https://api.example.com/health", "https://api.example.com/ready"]
    assert json.loads(capsys.readouterr().out)["ready"] is True


def test_cli_blocks_partial_backend_before_second_request():
    endpoints = []

    def mock_read(url, *, timeout):
        endpoints.append(url)
        return {"models": {}}

    with patch("scripts.check_web_inference_deployment.read_json", side_effect=mock_read):
        assert main(["--api-url", "https://api.example.com"]) == 1
    assert endpoints == ["https://api.example.com/health"]
