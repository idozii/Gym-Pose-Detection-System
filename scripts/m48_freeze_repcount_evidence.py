"""M48: freeze M46/M47 publisher TRAIN evidence and a reproducible source fingerprint.

NO TEST video/annotation access, NO training, NO new model selection. This
strictly compares the owner's existing M46 and M47 JSON files and records
the SHA256 of those exact bytes plus all code paths governing pose
extraction, selection, detection, matching and aggregation.

A frozen manifest is an audit aid, NOT an official TEST readiness
certificate, a subject-disjointness check or publisher license check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from scripts.m41_repcount_train_cadence import CONDITIONS
from scripts.m45_train_relative_motion_benchmark import (
    PROTOCOL as M45_PROTOCOL, EXPECTED_RULE
)
from scripts.m47_train_clip_bootstrap import (
    PROTOCOL as M47_PROTOCOL, BOOTSTRAP_SEED, BOOTSTRAP_REPEATS,
    SOURCE_RANK_OFFSET, EXPECTED_CLIPS
)

PROTOCOL = "m48-research-freeze-m46-m47-train-evidence-v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_M46 = Path("data/processed/m46_train_frozen_replication.json")
DEFAULT_M47 = Path("data/processed/m47_train_replication_bootstrap.json")
DEFAULT_REPORT = Path("data/processed/m48_train_evidence_freeze.json")
COMPARE_METHODS = ("low_to_high", "relative_excursion")
BOOT_METRICS = (
    "event_f1", "mean_absolute_count_error",
    "false_positive_events_per_video"
)
SOURCES = (
    "scripts/m37_repcount_tar.py",
    "scripts/m37_repcount_video_benchmark.py",
    "scripts/m39_repcount_train_baselines.py",
    "scripts/m41_repcount_train_cadence.py",
    "scripts/m43_repcount_train_error_audit.py",
    "scripts/m43_temporal_error_schema.py",
    "scripts/m44_train_boundary_angle_audit.py",
    "scripts/m44_boundary_angle_schema.py",
    "scripts/m45_train_relative_motion_benchmark.py",
    "scripts/m47_train_clip_bootstrap.py",
    "web/src/rep_counter.js",
    "web/scripts/m37_eval_repcount_trace.js",
    "web/scripts/m39_compare_train_trace.js",
    "web/scripts/m41_compare_train_cadence.js",
    "web/scripts/m43_analyse_train_errors.js",
    "web/scripts/m44_train_boundary_audit.js",
    "web/scripts/m45_compare_train_relative.js"
)

# Anchor to the owner's ALREADY OBSERVED, frozen fresh-source M46
# counts, not to a new interpretation chosen during M48.
M46_OWNER_COUNTS = {
    "dense_200ms": {
        "low_to_high": (117, 7, 92, 4.45),
        "relative_excursion": (129, 8, 80, 3.8)
    },
    "subsample_400ms": {
        "low_to_high": (94, 6, 115, 5.45),
        "relative_excursion": (121, 6, 88, 4.2)
    }
}
# Owner M47 5000 deterministic paired-video percentile intervals.
M47_OWNER_INTERVALS = {
    "dense_200ms": {
        "event_f1": (0.0018466060757148584, 0.08492339106681474),
        "mean_absolute_count_error": (-1.25, -0.10000000000000009),
        "false_positive_events_per_video": (-0.2, 0.30000000000000004)
    },
    "subsample_400ms": {
        "event_f1": (0.0313274965461259, 0.19067633767044562),
        "mean_absolute_count_error": (-2.45, -0.20000000000000018),
        "false_positive_events_per_video": (-0.2, 0.25000000000000006)
    }
}


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _numeric_close(actual, expected, label):
    if (type(actual) not in (float, int) or
        not math.isfinite(actual) or
        not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10)):
        raise ValueError("M48 owner-observed value mismatch: " + label)


def _nonneg_int(v, label):
    if type(v) is not int or v < 0:
        raise ValueError("M48 invalid integer metric " + label)
    return v


def _score_data(entry, expected, label):
    if not isinstance(entry, dict):
        raise ValueError("M48 missing method metrics " + label)
    tp, fp, fn, mae = expected
    for key, value in (
        ("event_tp", tp), ("event_fp", fp), ("event_fn", fn),
        ("annotated_reps", 209), ("videos", 20),
        ("predicted_reps", tp + fp)
    ):
        if _nonneg_int(entry.get(key), label + " " + key) != value:
            raise ValueError("M48 owner-observed count mismatch: " + label + " " + key)
    _numeric_close(entry.get("mean_absolute_count_error"), mae, label + " MAE")
    f1 = 2 * tp / (2 * tp + fp + fn)
    _numeric_close(entry.get("event_f1"), f1, label + " event F1")


def validate_evidence(m46: dict, m47: dict, m46_bytes: bytes) -> dict:
    """Return small frozen claims only after exact cross-report verification."""
    if not isinstance(m46, dict) or not isinstance(m47, dict):
        raise ValueError("M48 requires two JSON report objects")
    cohort = m46.get("source_selection")
    frozen = m47.get("frozen_source")
    settings = m47.get("analysis_settings")
    if any(not isinstance(v, dict) for v in (cohort, frozen, settings)):
        raise ValueError("M48 requires source and bootstrap metadata")
    sha = m46.get("publisher_train_annotation_sha256")
    if (type(sha) is not str or len(sha) != 64 or
        any(ch not in "0123456789abcdef" for ch in sha)):
        raise ValueError("M48 publisher TRAIN annotation SHA256 missing")
    if (
        m46.get("protocol") != M45_PROTOCOL or
        m47.get("protocol") != M47_PROTOCOL or
        m46.get("source_split") != "train" or
        m47.get("source_split") != "train" or
        m46.get("study_type") !=
            "prespecified_unseen_train_source_file_replication" or
        m46.get("relative_excursion_rule") != EXPECTED_RULE or
        m47.get("rule_unchanged") != EXPECTED_RULE or
        m46.get("annotated_reps") != 209 or
        m46.get("videos_selected") != EXPECTED_CLIPS or
        m46.get("videos_evaluated") != EXPECTED_CLIPS or
        m46.get("videos_excluded") != 0 or
        m47.get("publisher_train_annotation_sha256") != sha or
        m47.get("original_m46_report_sha256") != _sha(m46_bytes) or
        cohort.get("policy") != "stable_hash" or
        cohort.get("seed") != 2026 or
        cohort.get("source_hash_rank_offset") != SOURCE_RANK_OFFSET or
        cohort.get("limit") != EXPECTED_CLIPS or
        cohort.get("source_clip_filename_disjoint_from_m40_m45") is not True or
        cohort.get("subject_or_view_disjointness_verified") is not False or
        cohort.get("pose_observation_target_ms") != 200 or
        cohort.get("400ms_subsample_shares_dense_tracker_history") is not True or
        frozen.get("selection_policy") != "stable_hash" or
        frozen.get("selection_seed") != 2026 or
        frozen.get("hash_rank_offset") != SOURCE_RANK_OFFSET or
        frozen.get("videos_evaluated") != EXPECTED_CLIPS or
        frozen.get("videos_excluded") != 0 or
        frozen.get("annotated_gt_reps") != 209 or
        frozen.get("source_filenames_disjoint_from_original_first_20") is not True or
        frozen.get("subject_and_session_disjointness_verified") is not False or
        settings.get("paired_cluster_unit") != "source_video" or
        settings.get("bootstrap_method") != "percentile" or
        settings.get("quantiles") != [0.025, 0.975] or
        settings.get("draws") != BOOTSTRAP_REPEATS or
        settings.get("seed") != BOOTSTRAP_SEED or
        settings.get("difference_orientation") !=
            "relative_excursion_minus_low_to_high" or
        settings.get("two_sided_test_or_p_value_performed") is not False or
        m46.get("no_validation_or_test_access") is not True or
        m47.get("validation_and_test_untouched") is not True or
        m46.get("deployed_rep_counter_unchanged") is not True or
        m47.get("deployed_m34_unchanged") is not True
    ):
        raise ValueError("M48 frozen TRAIN source/protocol/reference mismatch")
    a = m46.get("conditions")
    b = m47.get("recomputed_frozen_m46_metrics")
    bootstrap = m47.get("paired_clip_bootstrap")
    if any(not isinstance(x, dict) or set(x) != set(CONDITIONS) for x in (a,b,bootstrap)):
        raise ValueError("M48 requires exact original two cadence conditions")
    supported = {}
    for cond in CONDITIONS:
        methods = a[cond].get("methods")
        repeated = b[cond]
        intervals = bootstrap[cond]
        if (not isinstance(methods, dict) or
            not isinstance(repeated, dict) or
            not isinstance(intervals, dict) or
            set(intervals) != set(BOOT_METRICS)):
            raise ValueError("M48 original and bootstrap method fields unavailable")
        for method in COMPARE_METHODS:
            expected = M46_OWNER_COUNTS[cond][method]
            _score_data(methods.get(method), expected, cond + " M46 " + method)
            _score_data(repeated.get(method), expected, cond + " M47 " + method)
            for countfield in ("event_tp", "event_fp", "event_fn", "videos",
                               "annotated_reps", "predicted_reps"):
                if methods[method][countfield] != repeated[method][countfield]:
                    raise ValueError("M48 exact M46/M47 event parity mismatch")
        low = methods["low_to_high"]
        relative = methods["relative_excursion"]
        expected_delta = {
            "event_f1": relative["event_f1"] - low["event_f1"],
            "mean_absolute_count_error": (
                relative["mean_absolute_count_error"] -
                low["mean_absolute_count_error"]
            ),
            "false_positive_events_per_video": (
                (relative["event_fp"] - low["event_fp"]) / EXPECTED_CLIPS
            )
        }
        supported[cond] = {}
        for metric in BOOT_METRICS:
            observed = intervals[metric]
            if not isinstance(observed, dict):
                raise ValueError("M48 missing bootstrap statistic " + metric)
            _numeric_close(
                observed.get("observed_relative_minus_low_to_high"),
                expected_delta[metric], cond + " delta " + metric
            )
            ci = observed.get("percentile_95_interval")
            lower, upper = M47_OWNER_INTERVALS[cond][metric]
            if not isinstance(ci, list) or len(ci) != 2:
                raise ValueError("M48 percentile bootstrap interval unavailable")
            _numeric_close(ci[0], lower, cond + " lower CI " + metric)
            _numeric_close(ci[1], upper, cond + " upper CI " + metric)
            if ci[0] > ci[1]:
                raise ValueError("M48 inverted bootstrap bounds")
            direction_keys = (
                "bootstrap_draws_strictly_above_zero",
                "bootstrap_draws_equal_to_zero",
                "bootstrap_draws_strictly_below_zero"
            )
            draws = [_nonneg_int(observed.get(k), k) for k in direction_keys]
            if sum(draws) != BOOTSTRAP_REPEATS:
                raise ValueError("M48 bootstrap directional draws do not sum to 5000")
            supported[cond][metric] = {
                "observed_relative_minus_low_to_high": expected_delta[metric],
                "video_bootstrap_percentile_95": [lower, upper]
            }
    return supported


def _fingerprint_sources(repository_root: Path) -> dict:
    if not repository_root.is_dir():
        raise ValueError("M48 repository root missing")
    checksums = {}
    for path in SOURCES:
        file = repository_root / path
        if not file.is_file() or file.is_symlink():
            raise ValueError("M48 research source file unavailable: " + path)
        checksums[path] = _sha(file.read_bytes())
    return checksums


def _revision(repository_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root, capture_output=True,
            text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("M48 requires local Git repository HEAD") from exc
    revision = proc.stdout.strip()
    if proc.returncode or len(revision) != 40 or any(
        char not in "0123456789abcdef" for char in revision
    ):
        raise ValueError("M48 cannot fingerprint Git HEAD")
    return revision


def run(*, m46_path: Path, m47_path: Path, report_path: Path,
        repository_root: Path = ROOT) -> dict:
    if report_path.exists():
        raise FileExistsError("M48 existing freeze report refuses overwrite")
    if not m46_path.is_file() or not m47_path.is_file():
        raise FileNotFoundError("M48 requires existing owner's original M46 AND M47 report")
    m46_bytes = m46_path.read_bytes()
    m47_bytes = m47_path.read_bytes()
    m46 = json.loads(m46_bytes)
    m47 = json.loads(m47_bytes)
    supported = validate_evidence(m46, m47, m46_bytes)
    source_hashes = _fingerprint_sources(repository_root)
    revision = _revision(repository_root)
    report = {
        "protocol": PROTOCOL,
        "purpose": "evidence_and_source_fingerprint_BEFORE_ANY_TEST_access",
        "source_split": "train",
        "owner_observed_evidence_locked": True,
        "report_sha256": {
            "original_m46_train_replication": _sha(m46_bytes),
            "original_m47_train_paired_bootstrap": _sha(m47_bytes)
        },
        "publisher_train_annotation_sha256": m46["publisher_train_annotation_sha256"],
        "git_head_commit_at_freeze": revision,
        "source_code_sha256": source_hashes,
        "known_protocol": {
            "training_cohort_rank_offset": SOURCE_RANK_OFFSET,
            "cohort_video_count": EXPECTED_CLIPS,
            "ground_truth_reps": 209,
            "selected_method": "relative_excursion",
            "control_method": "low_to_high",
            "frozen_relative_rule": EXPECTED_RULE,
            "primary_diagnostic_cadence": "dense_200ms",
            "secondary_cadence": "subsample_400ms",
            "source_split": "train",
            "original_gt_event_matcher": "m37_one_to_one_gt_end_tolerance",
            "ranking_seed": 2026,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_REPEATS,
            "bootstrap_unit": "source_video",
            "single_predeclared_test_execution_permitted_by_this_script": False
        },
        "reconciled_owner_M46_M47_deltas": supported,
        "unresolved_requirements_before_claiming_external_validation": [
            "Confirm permission/provenance of publisher RepCount-A archive and official annotation schema.",
            "Independently verify subject/session/view overlap across splits before asserting subject-disjoint performance.",
            "Define and permanently freeze future TEST cohort, unit, matching, exclusion and uncertainty policies before running TEST.",
            "Treat validation clips already examined in M38 as not untouched independent validation.",
            "Provide properly annotated negative controls from an authorized source; publisher TRAIN Squat count-zero cohort is absent.",
            "Confirm Python MediaPipe offline tracking is not equal to real-time browser model without parity testing.",
            "No unseen TEST accuracy, externally replicated conference evidence, anatomical squat form or injury-safety claims."
        ],
        "test_or_validation_media_scanned": False,
        "creates_test_evaluation": False,
        "manually_signed_or_officially_audited": False
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    return report


def verify_existing_lock(*, lock_path: Path, m46_path: Path,
                         m47_path: Path, repository_root: Path = ROOT) -> dict:
    """Read-only verification; refuse changed reports OR frozen source bytes."""
    if not lock_path.is_file():
        raise FileNotFoundError("M48 lock JSON does not exist")
    lock = json.loads(lock_path.read_bytes())
    if not isinstance(lock, dict) or lock.get("protocol") != PROTOCOL:
        raise ValueError("M48 lock protocol mismatch")
    if not m46_path.is_file() or not m47_path.is_file():
        raise FileNotFoundError("M48 verify requires original M46/M47 JSONs")
    m46_bytes = m46_path.read_bytes()
    m47_bytes = m47_path.read_bytes()
    if lock.get("report_sha256") != {
        "original_m46_train_replication": _sha(m46_bytes),
        "original_m47_train_paired_bootstrap": _sha(m47_bytes)
    }:
        raise ValueError("M48 owner report SHA256 changed since freeze")
    original = json.loads(m46_bytes)
    bootstrap = json.loads(m47_bytes)
    validate_evidence(original, bootstrap, m46_bytes)
    if lock.get("source_code_sha256") != _fingerprint_sources(repository_root):
        raise ValueError("M48 frozen research source-code SHA256 changed since lock")
    if (lock.get("source_split") != "train" or
        lock.get("test_or_validation_media_scanned") is not False or
        lock.get("creates_test_evaluation") is not False or
        lock.get("owner_observed_evidence_locked") is not True):
        raise ValueError("M48 lock evidence/protocol metadata invalid")
    return {
        "protocol": PROTOCOL,
        "verified_original_report_bytes": True,
        "verified_frozen_source_bytes": True,
        "original_git_head_at_lock": lock["git_head_commit_at_freeze"],
        "source_paths_verified": len(SOURCES),
        "test_or_validation_media_scanned": False
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m46-report", type=Path, default=DEFAULT_M46)
    parser.add_argument("--m47-report", type=Path, default=DEFAULT_M47)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verify-freeze", action="store_true",
                        help="Read-only check that M46, M47 and frozen source hashes remain identical")
    args = parser.parse_args(argv)
    if args.verify_freeze:
        try:
            verified = verify_existing_lock(
                lock_path=args.report, m46_path=args.m46_report,
                m47_path=args.m47_report
            )
        except (OSError, ValueError, TypeError, KeyError,
                json.JSONDecodeError) as exc:
            print("M48 freeze verification refused: " + str(exc), file=sys.stderr)
            return 1
        print(json.dumps(verified, indent=2))
        return 0
    try:
        result = run(
            m46_path=args.m46_report, m47_path=args.m47_report,
            report_path=args.report
        )
    except (OSError, ValueError, TypeError, KeyError,
            json.JSONDecodeError) as exc:
        print("M48 freeze refused: " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps({
        "protocol": result["protocol"],
        "git_head_commit_at_freeze": result["git_head_commit_at_freeze"],
        "source_code_paths_locked": len(result["source_code_sha256"]),
        "source_split": result["source_split"],
        "owner_observed_evidence_locked": result["owner_observed_evidence_locked"],
        "test_or_validation_media_scanned": result["test_or_validation_media_scanned"],
        "report_path": str(args.report)
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
