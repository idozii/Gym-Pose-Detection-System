"""Regression gates for opt-in paid Render model backend blueprint.

No Render APIs, local model files, source TEST clips, or remote servers.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_blueprint_requires_private_paid_disk_and_explicit_owner_trust():
    config = yaml.safe_load((ROOT / "render.yaml").read_text())
    assert isinstance(config, dict)
    assert len(config["services"]) == 1
    service = config["services"][0]
    assert service["type"] == "web"
    assert service["runtime"] == "docker"
    assert service["region"] == "singapore"
    assert service["plan"] == "1c-2g"
    assert service["repo"] == "https://github.com/idozii/Gym-Pose-Detection-System"
    assert service["branch"] == "main"
    assert service["dockerfilePath"] == "./backend/Dockerfile"
    assert service["dockerContext"] == "."
    assert service["autoDeployTrigger"] == "off"
    assert service["healthCheckPath"] == "/health"
    assert service["numInstances"] == 1
    assert service["disk"]["mountPath"] == "/private-models"
    assert service["disk"]["sizeGB"] == 1
    variables = {var["key"]: var for var in service["envVars"]}
    assert set(variables) == {
        "POSE_MODEL_ROOT", "POSE_WEB_ORIGINS", "POSE_TRUST_LOCAL_ARTIFACTS"
    }
    assert variables["POSE_MODEL_ROOT"]["value"] == "/private-models"
    assert variables["POSE_WEB_ORIGINS"]["value"] == (
        "https://gym-pose-detection-system.vercel.app"
    )
    assert variables["POSE_TRUST_LOCAL_ARTIFACTS"] == {
        "key": "POSE_TRUST_LOCAL_ARTIFACTS", "sync": False,
    }
    # No browser-embedded model URL, credentials, source data or auto-provisioned models.
    assert "VITE_INFERENCE_API_URL" not in variables
    assert "artifacts/" not in (ROOT / "render.yaml").read_text()


def test_backend_dockerfile_supports_render_ssh_without_root_execution():
    docker = (ROOT / "backend" / "Dockerfile").read_text()
    assert "COPY src /app/src" in docker
    assert "COPY artifacts" not in docker
    assert "COPY models" not in docker
    assert "USER pose:pose" in docker
    assert "--create-home --shell /bin/bash pose" in docker
    assert "chmod 700 /home/pose/.ssh" in docker
    assert "PORT:-8000" in docker
    assert "--no-access-log" in docker
