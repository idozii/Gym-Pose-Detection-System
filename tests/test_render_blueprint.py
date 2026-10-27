"""Guardrail tests for strictly FREE Render + private six-model downloads."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_render_blueprint_has_zero_paid_resources_and_trust_is_disabled():
    config = yaml.safe_load((ROOT / "render.yaml").read_text())
    assert len(config["services"]) == 1
    service = config["services"][0]
    assert service["type"] == "web"
    assert service["runtime"] == "docker"
    assert service["region"] == "singapore"
    assert service["plan"] == "free"
    assert service["repo"] == "https://github.com/idozii/Gym-Pose-Detection-System"
    assert service["branch"] == "main"
    assert service["dockerfilePath"] == "./backend/Dockerfile"
    assert service["dockerContext"] == "."
    assert service["autoDeployTrigger"] == "off"
    assert service["healthCheckPath"] == "/health"
    assert service["numInstances"] == 1
    assert "disk" not in service
    variables = {item["key"]: item for item in service["envVars"]}
    assert variables["POSE_MODEL_ROOT"]["value"] == "/tmp/gym-pose-models"
    assert variables["POSE_WEB_ORIGINS"]["value"] == (
        "https://gym-pose-detection-system.vercel.app"
    )
    assert variables["POSE_TRUST_LOCAL_ARTIFACTS"]["value"] == "0"
    for key in (
        "POSE_HF_REPO_ID", "POSE_HF_REVISION",
        "POSE_HF_MANIFEST_SHA256", "HF_TOKEN",
    ):
        assert variables[key] == {"key": key, "sync": False}
    assert "VITE_INFERENCE_API_URL" not in variables


def test_free_dockerfile_never_embeds_owner_models_or_ssh():
    docker = (ROOT / "backend" / "Dockerfile").read_text()
    assert "COPY src /app/src" in docker
    assert "COPY artifacts" not in docker
    assert "COPY models" not in docker
    assert "USER pose:pose" in docker
    assert "private_model_bootstrap" in docker
    # The owner serialized the six RandomForest models with sklearn 1.8.0.
    assert "scikit-learn==1.8.0" in docker
    assert "PORT:-8000" in docker
    assert "--no-access-log" in docker
    assert "--shell /usr/sbin/nologin" in docker
    assert "/private-models" not in docker
