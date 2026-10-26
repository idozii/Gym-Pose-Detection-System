#!/usr/bin/env python3
"""Read-only software release audit for the Fitness-AQA research prototype.

This is NOT an independent evaluation and never opens publisher TEST data,
loads joblib/pickle, uses a camera, or accesses a network. It inspects
tracked Git paths, package source contracts, and release documentation.
A passing audit means software source/provenance hygiene passed, NOT that
the classifiers are accurate, unbiased, or safe for exercise coaching.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

from gym_pose.model_registry import DEFAULT_PATHS, KNOWN_TASKS, _contract

REQUIRED_SCRIPTS = (
    "scripts/extract_fitness_aqa_ohp_tasks.py",
    "scripts/benchmark_fitness_aqa_ohp_tasks.py",
    "scripts/assess_fitness_aqa_ohp_seed_stability.py",
    "scripts/train_fitness_aqa_squat_source_errors.py",
    "scripts/benchmark_fitness_aqa_barbell_row_torso4.py",
    "scripts/check_fitness_aqa_model_registry.py",
    "scripts/run_fitness_aqa_research_viewer.py",
)
REQUIRED_DOCUMENTS = (
    "LICENSE",
    "README.md",
    "docs/M17_RESEARCH_REPORT_AND_RELEASE_READINESS.md",
    "docs/FITNESS_AQA_OHP_TASK_SPECIFIC_FEATURES.md",
    "docs/FITNESS_AQA_OHP_STATIC_TEMPORAL_RF_M13D.md",
    "docs/FITNESS_AQA_OHP_MULTISEED_M13E.md",
)
EXPECTED_FEATURE_COUNTS = {
    ("squat", "error_knees_forward"): 103,
    ("squat", "error_knees_inward"): 103,
    ("ohp", "error_elbows"): 30,
    ("ohp", "error_knees"): 30,
    ("barbell_row", "torso_angle"): 20,
    ("barbell_row", "lumbar"): 20,
}
HISTORICAL_LEGACY_BINARIES = frozenset({
    "models/squat_form_rf.joblib",
    "models/yolov8s.pt",
})
MEDIA_SUFFIXES = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi", ".mov",
    ".mkv", ".zip", ".7z", ".tar", ".pt", ".joblib", ".pkl", ".onnx",
})
DERIVED_PREFIXES = (
    "artifacts/", "data/raw/", "data/external/", "data/processed/",
    "data/cache/", "datasets/", "Fitness-AQA_dataset_release/",
)


def tracked_files_from_git(root: Path) -> list[str]:
    """List actual repository-tracked filenames; do not scan ignored local data."""
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("Unable to inspect repository git index")
    return sorted({
        p.decode("utf-8", errors="replace").replace("\\", "/")
        for p in proc.stdout.split(b"\0") if p
    })


def audit_tracked_paths(paths: Iterable[str]) -> dict:
    """Prevent committing research datasets and new model binary artifacts.

    The two *already tracked* legacy binaries are reported honestly as
    release caveats, rather than silently deleting historical notebook assets.
    """
    blocked: set[str] = set()
    historical: set[str] = set()
    for raw in paths:
        p = str(raw).replace("\\", "/")
        if p.startswith("./"):
            p = p[2:]
        if not p or p.startswith("../") or "/../" in p:
            blocked.add(raw)
            continue
        if p in HISTORICAL_LEGACY_BINARIES:
            historical.add(p)
            continue
        # Empty marker files preserve a directory's structure in Git.
        # They contain NO user media, derived records or source TEST data.
        if p.endswith("/.gitkeep") and (
            p.startswith(DERIVED_PREFIXES) or p.startswith("data/features/")
        ):
            continue
        if (p == "data/manifest.csv" or
            any(part.endswith(".egg-info") for part in p.split("/")) or
            p.startswith(DERIVED_PREFIXES) or
            p.startswith("data/features/") and not p.endswith("/.gitkeep") or
            p.startswith("models/") and Path(p).suffix.lower() in MEDIA_SUFFIXES or
            p.startswith("data/") and Path(p).suffix.lower() in MEDIA_SUFFIXES or
            Path(p).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".zip", ".7z"}):
            blocked.add(p)
    return {
        "blocked_new_private_or_model_artifacts": sorted(blocked),
        "previously_tracked_legacy_model_binaries": sorted(historical),
        "tracked_file_count": len(set(paths)),
        "passed": not blocked,
    }


def audit_task_contracts() -> dict:
    """Validate named source-label versions without loading any .joblib files."""
    entries = {}
    problems = []
    if len(KNOWN_TASKS) != 6 or set(KNOWN_TASKS) != set(EXPECTED_FEATURE_COUNTS):
        problems.append("Expected exactly six original source-error targets")
    if set(DEFAULT_PATHS) != set(KNOWN_TASKS):
        problems.append("Registry model-path mapping does not cover the exact source targets")
    for exercise, target in KNOWN_TASKS:
        task = f"{exercise}/{target}"
        try:
            contract = _contract(exercise, target)
            if contract is None:
                raise ValueError("Missing contract")
            count = len(contract.feature_names)
            if (count != EXPECTED_FEATURE_COUNTS[(exercise, target)]
                or len(set(contract.feature_names)) != count):
                raise ValueError("Wrong feature count or repeated feature names")
            if not contract.path.as_posix().startswith("artifacts/"):
                raise ValueError("Current registry output outside ignored artifacts/")
            if contract.variant not in (None, "static"):
                raise ValueError("Unexpected nonstatic single-frame variant")
            if not contract.schema_version or not contract.threshold_key:
                raise ValueError("Missing source version/threshold provenance")
            entries[task] = {
                "schema": contract.schema_version,
                "feature_count": count,
                "artifact_path": contract.path.as_posix(),
                "artifact_deserialized": False,
            }
        except (KeyError, ValueError, TypeError) as exc:
            problems.append(f"{task}: {exc}")
    return {
        "passed": not problems,
        "task_contract_count": len(entries),
        "models_loaded": 0,
        "contracts": entries,
        "errors": problems,
    }


def audit_research_files(root: Path) -> dict:
    """Check research source/methodology and explicit historic TEST caveat."""
    required = (*REQUIRED_SCRIPTS, *REQUIRED_DOCUMENTS)
    missing = sorted(p for p in required if not (root / p).is_file())
    caveat_ok = False
    disclaimer_ok = False
    report = root / "docs/M17_RESEARCH_REPORT_AND_RELEASE_READINESS.md"
    readme = root / "README.md"
    if report.is_file():
        text = report.read_text(encoding="utf-8")
        caveat_ok = ("TEST" in text and "historical" in text.lower()
                     and "not" in text.lower()
                     and "subject" in text.lower())
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        disclaimer_ok = bool(re.search(
            r"research.only|research prototype|research viewer", text, re.I
        )) and bool(re.search(r"not a validated|not a .*coach", text, re.I))
    return {
        "passed": not missing and caveat_ok and disclaimer_ok,
        "missing_reproducibility_files": missing,
        "historical_test_exposure_caveat_present": caveat_ok,
        "research_only_readme_disclaimer_present": disclaimer_ok,
    }


def audit_release(root: str | Path, *, tracked_paths: Iterable[str] | None = None) -> dict:
    """Run a static source-integrity gate, not a data/test/model accuracy gate."""
    root = Path(root).resolve()
    if tracked_paths is None:
        tracked_paths = tracked_files_from_git(root)
    tracked = audit_tracked_paths(list(tracked_paths))
    contracts = audit_task_contracts()
    source = audit_research_files(root)
    pass_software_gate = all(
        check["passed"] for check in (tracked, contracts, source))
    return {
        "status": ("research_software_release_audit_PASS"
                   if pass_software_gate else "research_software_release_audit_BLOCKED"),
        "scope": "experimental_research_software_source_integrity_NOT_ACCURACY",
        "research_only": True,
        "safe_for_fitness_coaching_claims": False,
        "publisher_test_media_opened_by_this_audit": False,
        "historical_publisher_test_pristine_claim_supported": False,
        "trusted_local_artifacts_deserialized": False,
        "camera_accessed": False,
        "checks": {
            "tracked_data_and_artifact_hygiene": tracked,
            "six_original_source_error_contracts": contracts,
            "documentation_and_reproducibility": source,
        },
        "release_notes": [
            "Passing source audit is not independent subject/athlete validation",
            "Source validation has been repeatedly consulted for exploratory evaluation",
            "Historical early pilot used some publisher TEST videos; see M17 report",
            "Two legacy notebook binaries remain tracked and are not part of the six trusted research model bundles",
            "OHP/Barbell Row live inference and external camera generalization not independently tested",
            "Do not deploy autonomous exercise correctness, medical risk or injury-prevention advice",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root (read-only); defaults to current source checkout")
    args = parser.parse_args()
    try:
        result = audit_release(args.root)
    except (RuntimeError, OSError, UnicodeError) as err:
        parser.exit(2, f"Release audit could not complete: {err}\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"].endswith("_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
