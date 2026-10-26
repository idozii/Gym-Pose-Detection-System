"""Offline M17B release-gate tests; never load pickle, media or user files."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import audit_research_release as audit


def mock_repo(root: Path, *, caveat=True, disclaimer=True):
    for path in (*audit.REQUIRED_SCRIPTS, *audit.REQUIRED_DOCUMENTS):
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("synthetic source file\n", encoding="utf-8")
    if caveat:
        (root / "docs/M17_RESEARCH_REPORT_AND_RELEASE_READINESS.md").write_text(
            "The historical publisher TEST pilot was examined; it is not "
            "proven subject-independent.\n", encoding="utf-8")
    if disclaimer:
        (root / "README.md").write_text(
            "This is a research-only prototype, not a validated "
            "fitness coach.\n", encoding="utf-8")
    return root


def test_release_gate_passes_only_as_software_audit(tmp_path):
    root = mock_repo(tmp_path)
    report = audit.audit_release(root, tracked_paths=[
        "README.md", "LICENSE",
        "data/manifest.example.csv", "data/features/.gitkeep",
        "data/raw/.gitkeep",
        "models/squat_form_rf.joblib", "models/yolov8s.pt",
    ])
    assert report["status"] == "research_software_release_audit_PASS"
    assert report["research_only"]
    assert not report["safe_for_fitness_coaching_claims"]
    assert not report["historical_publisher_test_pristine_claim_supported"]
    assert not report["publisher_test_media_opened_by_this_audit"]
    assert not report["trusted_local_artifacts_deserialized"]
    assert not report["camera_accessed"]
    checks = report["checks"]
    assert checks["six_original_source_error_contracts"]["task_contract_count"] == 6
    assert checks["tracked_data_and_artifact_hygiene"][
        "previously_tracked_legacy_model_binaries"] == [
            "models/squat_form_rf.joblib", "models/yolov8s.pt",
        ]
    assert checks["tracked_data_and_artifact_hygiene"][
        "blocked_new_private_or_model_artifacts"] == []


@pytest.mark.parametrize("forbidden", [
    "artifacts/final_model.joblib",
    "data/external/fitness_aqa/videos.zip",
    "data/raw/private_squat.jpg",
    "data/features/fitness_aqa_squat_train_pose.csv",
    "data/processed/session.parquet",
    "datasets/fitness_aqa/videos.zip",
    "Fitness-AQA_dataset_release/a.mp4",
    "models/new_huge_model.pt",
    "data/temp/private.png",
    "src/secret.zip",
    "../outside.csv",
])
def test_release_gate_refuses_new_tracked_research_data(tmp_path, forbidden):
    report = audit.audit_release(
        mock_repo(tmp_path), tracked_paths=["README.md", forbidden])
    assert report["status"] == "research_software_release_audit_BLOCKED"
    assert forbidden in report["checks"]["tracked_data_and_artifact_hygiene"][
        "blocked_new_private_or_model_artifacts"]


def test_readme_disclaimer_and_test_history_are_mandatory(tmp_path):
    root = mock_repo(tmp_path, caveat=False)
    fail = audit.audit_release(root, tracked_paths=["README.md"])
    assert fail["status"].endswith("_BLOCKED")
    assert not fail["checks"]["documentation_and_reproducibility"][
        "historical_test_exposure_caveat_present"]
    mock_repo(tmp_path, disclaimer=False)
    (root / "README.md").write_text(
        "Reliable and medically safe AI Gym Coach", encoding="utf-8")
    fail = audit.audit_release(root, tracked_paths=["README.md"])
    assert fail["status"].endswith("_BLOCKED")
    assert not fail["checks"]["documentation_and_reproducibility"][
        "research_only_readme_disclaimer_present"]


def test_missing_extractor_or_model_benchmark_blocks_reproducibility(tmp_path):
    root = mock_repo(tmp_path)
    (root / "scripts/benchmark_fitness_aqa_ohp_tasks.py").unlink()
    result = audit.audit_release(root, tracked_paths=["README.md"])
    assert result["status"].endswith("_BLOCKED")
    assert ("scripts/benchmark_fitness_aqa_ohp_tasks.py" in
            result["checks"]["documentation_and_reproducibility"][
                "missing_reproducibility_files"])


def test_six_contracts_match_actual_versioned_code_without_pickle():
    result = audit.audit_task_contracts()
    assert result["passed"]
    assert result["models_loaded"] == 0
    assert result["task_contract_count"] == 6
    assert {
        name: entry["feature_count"]
        for name, entry in result["contracts"].items()
    } == {
        "squat/error_knees_forward": 103,
        "squat/error_knees_inward": 103,
        "ohp/error_elbows": 30,
        "ohp/error_knees": 30,
        "barbell_row/torso_angle": 20,
        "barbell_row/lumbar": 20,
    }


def test_release_audit_does_not_recurse_into_user_ignored_artifacts(tmp_path):
    root = mock_repo(tmp_path)
    # These are NEVER opened or classified because git did not track them.
    leaked = root / "data/external/fitness_aqa/squat/test.mp4"
    leaked.parent.mkdir(parents=True)
    leaked.write_text("TEST_MEDIA_TRIPWIRE", encoding="utf-8")
    pickle = root / "artifacts/fitness_aqa_squat_source_rf/fake.joblib"
    pickle.parent.mkdir(parents=True)
    pickle.write_bytes(b"\x80\x04malicious pickle content")
    out = audit.audit_release(root, tracked_paths=["README.md"])
    assert out["status"].endswith("_PASS")
    assert out["publisher_test_media_opened_by_this_audit"] is False
    assert out["trusted_local_artifacts_deserialized"] is False
