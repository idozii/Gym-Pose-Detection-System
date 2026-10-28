"""M49: read-only pre-TEST Git provenance and preregistration gate.

Verify that the 17 source files snapshotted in the owner's M48 lock:
  1. still match byte-for-byte in the working tree (via M48 verification),
  2. match the corresponding tracked blobs at the original M48 Git commit,
  3. match the corresponding tracked blobs at today's Git HEAD.

M49 records a protocol for a future, separately authorized one-shot TEST
study but does NOT authorize it, touch any TEST/validation files, read a
publisher TAR or execute an evaluation.

All evidence values originate from M46/M47/M48 owner-local JSONs. This
is a software-provenance audit, NOT subject-independent data validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

from scripts import m48_freeze_repcount_evidence as m48

PROTOCOL = "m49-pretest-git-source-provenance-v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_M46 = Path("data/processed/m46_train_frozen_replication.json")
DEFAULT_M47 = Path("data/processed/m47_train_replication_bootstrap.json")
DEFAULT_M48 = Path("data/processed/m48_train_evidence_freeze.json")
DEFAULT_REPORT = Path("data/processed/m49_pretest_provenance.json")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _git_bytes(root: Path, *args: str) -> bytes:
    try:
        process = subprocess.run(
            ["git", *args], cwd=root, capture_output=True,
            timeout=20, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("M49 Git command could not complete") from exc
    if process.returncode:
        raise ValueError("M49 cannot verify frozen source at Git revision")
    return process.stdout


def verify_git_history(lock: dict, *, repository_root: Path = ROOT) -> dict:
    """Require byte-equivalent blobs at M48 lock commit and current HEAD.

    A newer Git HEAD is expected after adding M49. Thus we do NOT require
    current HEAD == M48 lock HEAD, but both must contain identical frozen
    source code for all 17 files.
    """
    if not isinstance(lock, dict) or lock.get("protocol") != m48.PROTOCOL:
        raise ValueError("M49 original M48 lock protocol mismatch")
    committed = lock.get("git_head_commit_at_freeze")
    fingerprints = lock.get("source_code_sha256")
    if (not isinstance(committed, str) or
        COMMIT_PATTERN.fullmatch(committed) is None or
        not isinstance(fingerprints, dict) or
        set(fingerprints) != set(m48.SOURCES)):
        raise ValueError("M49 M48 lock commit or 17 source hashes missing")
    actual_root = Path(
        _git_bytes(repository_root, "rev-parse", "--show-toplevel")
        .decode("utf8").strip()
    ).resolve()
    if actual_root != repository_root.resolve():
        raise ValueError("M49 repository root is not current Git toplevel")
    current_head = (
        _git_bytes(repository_root, "rev-parse", "--verify", "HEAD")
        .decode("ascii").strip()
    )
    if COMMIT_PATTERN.fullmatch(current_head) is None:
        raise ValueError("M49 invalid Git HEAD")
    for path in m48.SOURCES:
        expected = fingerprints[path]
        if (not isinstance(expected, str) or
            COMMIT_PATTERN.fullmatch(expected) is None):
            raise ValueError("M49 invalid SHA256 of frozen source")
        current_path = repository_root / path
        if not current_path.is_file() or current_path.is_symlink():
            raise ValueError("M49 frozen source missing or symlinked: " + path)
        if _sha(current_path.read_bytes()) != expected:
            raise ValueError("M49 local research source differs from M48 lock: " + path)
        for revision in (committed, current_head):
            blob = _git_bytes(
                repository_root, "show", f"{revision}:{path}"
            )
            if _sha(blob) != expected:
                raise ValueError(
                    "M49 frozen research source differs from Git tracked "
                    + revision[:12] + ": " + path
                )
    return {
        "original_m48_git_commit": committed,
        "current_git_commit": current_head,
        "frozen_source_paths_checked": len(m48.SOURCES),
        "source_bytes_match_original_m48_git_commit": True,
        "source_bytes_match_current_git_commit": True,
        "local_source_bytes_match_m48_lock": True,
        "non_source_changes_since_m48_lock_allowed": True
    }


# Intent is registered while retaining the TEST embargo. This is a
# proposed research protocol, not authorization to execute any TEST.
PREREGISTRATION = {
    "study_goal": "One-time future authorized publisher TEST comparison of frozen M45 and M39",
    "test_execution_authorized_by_this_tool": False,
    "dataset_scope_for_this_tool": "owner-local M46/M47/M48 reports and tracked Git source files ONLY",
    "candidate": "M45 relative_excursion unchanged",
    "control": "M39 low_to_high unchanged",
    "source": "official publisher RepCount-A, exact TEST definition pending licensed-author verification",
    "future_test_source_selection": "ALL eligible official Squat TEST entries, with source split/path checks and no outcome-based selection",
    "future_primary_condition": "200ms offline Python MediaPipe pose track",
    "future_secondary_condition": "every-other 400ms pose observations sharing 200ms tracking history",
    "future_primary_metric": "micro event F1 using frozen M37 one-to-one GT-end matching",
    "future_required_companions": [
        "video-mean absolute count error",
        "micro event precision and recall",
        "FP unmatched events per source video",
        "TP, FP, FN, total publisher GT events and exclusions"
    ],
    "future_uncertainty_unit": "source participant if authoritative metadata verifies identity; otherwise video with dependence explicitly acknowledged",
    "future_uncertainty": "paired bootstrap with replacement, fixed 5000 draws and seed 2026; percentile 95%, without p-value claims",
    "future_no_test_adaptation": True,
    "future_test_runs_planned": 1,
    "freeze_before_future_test": [
        "source/archive provenance and use permissions",
        "publisher GT interval/video FPS parity",
        "participant or session overlap policy",
        "event matcher, model, pose pipeline and runtime",
        "exclusion policy, full-cohort definition and uncertainty unit",
        "precommitted response to missing/invalid source clips"
    ]
}

MANUAL_GATES = {
    "publisher_archive_origin_and_permission": "not_verified",
    "official_test_split_access_authorized": "not_authorized_by_this_script",
    "cross_split_subject_and_session_overlap": "not_verified",
    "publisher_GT_frame_video_time_parity": "not_verified",
    "python_offline_vs_browser_pose_parity": "not_verified",
    "negative_only_squat_controls": "unavailable_in_original_publisher_train",
    "M38_previously_examined_validation_is_not_unseen": "acknowledged"
}


def run(*, m46_path: Path, m47_path: Path, lock_path: Path,
        report_path: Path, repository_root: Path = ROOT,
        check_only: bool = False) -> dict:
    if not check_only and report_path.exists():
        raise FileExistsError("M49 existing provenance report refuses overwrite")
    if not lock_path.is_file():
        raise FileNotFoundError("M49 requires original verified M48 lock JSON")
    # Cross-report and local source checks are deliberately delegated to
    # the identical read-only M48 verification algorithm.
    original = m48.verify_existing_lock(
        lock_path=lock_path, m46_path=m46_path, m47_path=m47_path,
        repository_root=repository_root
    )
    if (original.get("verified_original_report_bytes") is not True or
        original.get("verified_frozen_source_bytes") is not True):
        raise ValueError("M49 needs valid original M48 evidence verification")
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    provenance = verify_git_history(lock, repository_root=repository_root)
    result = {
        "protocol": PROTOCOL,
        "purpose": "before_any_test_source_access",
        "source_split_used_by_m49": "train",
        "original_m48_lock_sha256": _sha(lock_bytes),
        "frozen_training_annotation_sha256": lock[
            "publisher_train_annotation_sha256"
        ],
        "source_git_provenance": provenance,
        "preregistered_future_test_protocol_without_execution": PREREGISTRATION,
        "unresolved_manual_gates": MANUAL_GATES,
        "research_source_provenance_check_passed": True,
        "external_test_readiness_approved": False,
        "official_test_executed_or_opened": False,
        "validation_or_test_media_scanned": False,
        "no_original_counter_modified": True,
        "not_a_signed_audit_or_external_validation": True
    }
    if not check_only:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m46-report", type=Path, default=DEFAULT_M46)
    parser.add_argument("--m47-report", type=Path, default=DEFAULT_M47)
    parser.add_argument("--m48-lock", type=Path, default=DEFAULT_M48)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate M48 and both Git commits without writing a report")
    args = parser.parse_args(argv)
    try:
        result = run(
            m46_path=args.m46_report, m47_path=args.m47_report,
            lock_path=args.m48_lock, report_path=args.report,
            check_only=args.check_only
        )
    except (OSError, ValueError, TypeError, KeyError,
            UnicodeDecodeError, json.JSONDecodeError) as exc:
        print("M49 pre-TEST provenance refused: " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps({
        "protocol": result["protocol"],
        "research_source_provenance_check_passed":
            result["research_source_provenance_check_passed"],
        "source_git_provenance": result["source_git_provenance"],
        "external_test_readiness_approved":
            result["external_test_readiness_approved"],
        "official_test_executed_or_opened": False,
        "report_written": not args.check_only
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
