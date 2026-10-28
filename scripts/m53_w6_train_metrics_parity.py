"""M53 + W6: locked TRAIN metrics and same-input browser-rule parity replay.

M53 adds per-video normalized count MAE, Off-By-One and exact count
accuracy to the original M46/M47 20-video publisher TRAIN replication.
W6 audits incremental M45 website rule vs the EXACT original frozen
M45 JS comparator on the SAME offline Python landmark samples, at both
nominal 200ms and shared-tracker 400ms.

Strictly verify owner-local M46/M47/M48 and original 17 frozen sources
at both Git commits via M49; reuse M47's fully verified replay with
reference M46 aggregates unchanged. No file from the dataset TEST or
previously inspected validation split is accessed. Not a MediaPipe
Python-vs-browser estimator or actual browser-runtime parity study.
Only aggregate reports saved; raw traces/GT frame events temporary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from scripts import m47_train_clip_bootstrap as m47
from scripts.m49_pretest_git_provenance import run as pretest_m49

PROTOCOL = "m53-frozen-train-count-metrics-v1"
W6_PROTOCOL = "w6-same-landmarks-live-m45-algorithmic-parity-v1"
ROOT = Path("data/external/m53_train_metric_parity")
REPORT = Path("data/processed/m53_train_normalized_count_metrics.json")
W6_REPORT = Path("data/processed/w6_train_same_landmark_parity.json")
SCRIPT_W6 = (Path(__file__).resolve().parents[1] /
             "web/scripts/w6_same_landmark_parity.js")
METHODS = ("low_to_high", "relative_excursion")
CONDITIONS = ("dense_200ms", "subsample_400ms")
N = 20
DRAWS = 5000
SEED = 2026


def metrics_for_videos(items):
    """Per-video ratio mean, exact/Off-By-One fraction, no per-clip output."""
    if not isinstance(items, list) or not items:
        raise ValueError("M53 requires nonempty paired TRAIN video results")
    samples = {name: [] for name in METHODS}
    for item in items:
        if not isinstance(item, dict) or set(item) != set(METHODS):
            raise ValueError("M53 requires paired frozen M39/M45 methods")
        gt = None
        for name in METHODS:
            row = item[name]
            if not isinstance(row, dict):
                raise ValueError("M53 method record must be an object")
            n, pred, abs_err = (row.get(k) for k in (
                "annotated_count", "predicted_count", "absolute_count_error"
            ))
            tp, fp, fn = (row.get(k) for k in (
                "event_tp", "event_fp", "event_fn"
            ))
            if (not all(type(x) is int for x in (
                    n, pred, abs_err, tp, fp, fn)) or
                n <= 0 or pred < 0 or abs_err < 0 or
                any(x < 0 for x in (tp, fp, fn)) or
                tp + fn != n or tp + fp != pred or
                abs_err != abs(pred - n)):
                raise ValueError("M53 invalid original positive-GT clip count")
            if gt is None:
                gt = n
            elif n != gt:
                raise ValueError("M53 paired methods disagree on publisher GT")
            samples[name].append((n, pred, abs_err))
    n_videos = len(items)
    result = {}
    for name, triples in samples.items():
        result[name] = {
            "videos": n_videos,
            "annotated_gt_reps": sum(t[0] for t in triples),
            "predicted_reps": sum(t[1] for t in triples),
            "mean_normalized_absolute_count_error": sum(
                err / gt for gt, _pred, err in triples
            ) / n_videos,
            "off_by_one_fraction": sum(
                err <= 1 for _gt, _pred, err in triples
            ) / n_videos,
            "exact_count_fraction": sum(
                err == 0 for _gt, _pred, err in triples
            ) / n_videos,
            "absolute_count_mae": sum(t[2] for t in triples) / n_videos,
            "videos_with_off_by_one_or_better": sum(
                err <= 1 for _gt, _pred, err in triples
            ),
            "videos_with_exact_count": sum(
                err == 0 for _gt, _pred, err in triples
            )
        }
    return result


FIELDS = (
    "mean_normalized_absolute_count_error",
    "off_by_one_fraction",
    "exact_count_fraction",
    "absolute_count_mae"
)


def paired_video_bootstrap(items, *, repeats=DRAWS, seed=SEED):
    """Paired VIDEO bootstrap, 95% percentile; no identity output."""
    if (len(items) != N or type(repeats) is not int or
        not 1 <= repeats <= 50000 or type(seed) is not int):
        raise ValueError("M53 requires original 20-video paired TRAIN cohort")
    rng = random.Random(seed)
    observed = metrics_for_videos(items)
    differences = {field: [] for field in FIELDS}
    for _ in range(repeats):
        draw = [items[rng.randrange(N)] for _ in range(N)]
        summary = metrics_for_videos(draw)
        for field in FIELDS:
            differences[field].append(
                summary["relative_excursion"][field] -
                summary["low_to_high"][field]
            )
    intervals = {}
    for field in FIELDS:
        values = sorted(differences[field])
        intervals[field] = {
            "candidate_minus_baseline": (
                observed["relative_excursion"][field] -
                observed["low_to_high"][field]
            ),
            "percentile_95_interval": [
                m47._percentile(values, .025),
                m47._percentile(values, .975)
            ]
        }
    return intervals


def invoke_w6(trace_path: Path, output_path: Path):
    """Local Node subprocess, W6 trace and original GT never saved to output."""
    try:
        completed = subprocess.run(
            ["node", str(SCRIPT_W6), str(trace_path), str(output_path)],
            capture_output=True, text=True, check=False, timeout=90
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("M53/W6 requires Node and W6 JS runner") from error
    if completed.returncode:
        raise RuntimeError("W6 same-input parity evaluator refused trace")
    result = json.loads(output_path.read_text(encoding="utf8"))
    if (result.get("protocol") != W6_PROTOCOL or
        result.get("source_split") != "train" or
        result.get("raw_frames_or_event_times_saved") is not False or
        result.get("browser_or_python_pose_parity_claimed") is not False or
        set(result.get("conditions", {})) != set(CONDITIONS)):
        raise ValueError("W6 provenance or aggregate schema changed")
    for name in CONDITIONS:
        row = result["conditions"][name]
        if (not isinstance(row, dict) or
            row.get("gate_sample_ms") != 200 or
            not all(type(row.get(k)) is int and row[k] >= 0 for k in (
                "source_frames", "gate_selected_frames",
                "original_m45_event_count",
                "browser_incremental_event_count",
                "direct_same_input_event_disagreements",
                "gate_reference_event_count",
                "gate_browser_event_count",
                "gate_same_input_event_disagreements",
                "gate_dropped_frames"
            )) or
            row["source_frames"] != row["gate_selected_frames"] +
            row["gate_dropped_frames"] or
            row.get("pose_estimator_parity_verified") is not False or
            row.get("webpage_runtime_parity_verified") is not False):
            raise ValueError("W6 expected same-input aggregate schema missing")
    return result


def summarize_w6(records):
    if not isinstance(records, list) or len(records) != N:
        raise ValueError("W6 expected all 20 TRAIN videos to be audited")
    summary = {}
    for condition in CONDITIONS:
        rows = [r["conditions"][condition] for r in records]
        fields = (
            "source_frames", "gate_selected_frames",
            "gate_dropped_frames", "original_m45_event_count",
            "browser_incremental_event_count",
            "direct_same_input_event_disagreements",
            "gate_reference_event_count", "gate_browser_event_count",
            "gate_same_input_event_disagreements"
        )
        total = {key: sum(row[key] for row in rows) for key in fields}
        total.update({
            "videos": N,
            "videos_with_direct_disagreement": sum(
                row["direct_same_input_event_disagreements"] != 0
                for row in rows
            ),
            "videos_with_gated_disagreement": sum(
                row["gate_same_input_event_disagreements"] != 0
                for row in rows
            ),
            "all_same_landmark_event_times_equal": all(
                row["direct_exact_event_parity"] is True for row in rows
            ),
            "all_gated_same_landmark_event_times_equal": all(
                row["gated_exact_event_parity"] is True for row in rows
            )
        })
        summary[condition] = total
    return summary


def run(*, archive: Path, root: Path, reference_path: Path,
        m47_path: Path, lock_path: Path, report_path: Path,
        w6_report_path: Path, replay_fn=None, parity_fn=None):
    if report_path.resolve() == w6_report_path.resolve():
        raise ValueError("M53 and W6 output paths must be different")
    if report_path.exists() or w6_report_path.exists():
        raise FileExistsError("M53/W6 report already exists; refusing overwrite")
    # Read-only verification of 17 frozen sources and owner M46/M47/M48 bytes.
    gate = pretest_m49(
        m46_path=reference_path, m47_path=m47_path, lock_path=lock_path,
        report_path=Path("data/processed/m49_pretest_provenance.json"),
        check_only=True
    )
    if (gate.get("research_source_provenance_check_passed") is not True or
        gate.get("official_test_executed_or_opened") is not False):
        raise ValueError("M53 requires original validated M49 research lock")
    original_rows = []
    parity_rows = []
    evaluator = replay_fn or m47.run
    w6 = parity_fn or invoke_w6

    def compare(trace_path: Path, output_path: Path):
        row = m47.compare_original_m45(trace_path, output_path)
        m47.validate_original_m45(row, row["annotated_rep_count"])
        original_rows.append(row)
        w6_output = output_path.with_name(output_path.stem + "_w6.json")
        parity_rows.append(w6(trace_path, w6_output))
        return row

    # M47 performs original TRAIN source extraction / score verification
    # against owner M46 and reruns the original M47 paired bootstrap.
    # The temporary M47 report is never published or used as new evidence.
    with tempfile.TemporaryDirectory(prefix="m53-private-report-") as temp:
        temp_report = Path(temp) / "original_m47_rereplay.json"
        replayed = evaluator(
            archive=archive, root=root, report_path=temp_report,
            reference_path=reference_path,
            compare_fn=compare,
            repeats=DRAWS, seed=SEED
        )
        if (replayed.get("source_split") != "train" or
            replayed.get("frozen_source", {}).get("videos_evaluated") != N or
            replayed.get("frozen_source", {}).get(
                "subject_and_session_disjointness_verified") is not False or
            replayed.get("validation_and_test_untouched") is not True or
            len(original_rows) != N or len(parity_rows) != N):
            raise ValueError("M53 original M47 replay or W6 coverage incomplete")

    all_methods = {}
    extended_ci = {}
    for condition in CONDITIONS:
        items = [m47._per_video(row, condition) for row in original_rows]
        methods = metrics_for_videos(items)
        # Count-MAE MUST reproduce the original M46 independently
        # verified point estimate, without tolerance or threshold tuning.
        original_methods = replayed["recomputed_frozen_m46_metrics"][condition]
        for name in METHODS:
            expected = original_methods[name]
            if (not math.isclose(methods[name]["absolute_count_mae"],
                                 expected["mean_absolute_count_error"],
                                 rel_tol=0, abs_tol=1e-10) or
                methods[name]["annotated_gt_reps"] !=
                expected["annotated_reps"] or
                methods[name]["predicted_reps"] !=
                expected["predicted_reps"] or
                methods[name]["videos"] != N or
                not math.isclose(methods[name]["exact_count_fraction"],
                                 expected["exact_count_fraction"],
                                 rel_tol=0, abs_tol=1e-10) or
                not math.isclose(methods[name]["off_by_one_fraction"],
                                 expected["within_one_fraction"],
                                 rel_tol=0, abs_tol=1e-10)):
                raise ValueError("M53 count MAE/GT does not match original M46")
        all_methods[condition] = methods
        extended_ci[condition] = paired_video_bootstrap(items)

    sha_m46 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    sha_m47 = hashlib.sha256(m47_path.read_bytes()).hexdigest()
    sha_m48 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    source_sha = replayed["publisher_train_annotation_sha256"]
    report = {
        "protocol": PROTOCOL,
        "source_split": "train",
        "cohort": "frozen_replication_filename_hash_ranks_21_through_40",
        "videos_evaluated": N,
        "annotated_gt_reps": 209,
        "video_exclusions": 0,
        "original_m46_report_sha256": sha_m46,
        "original_m47_report_sha256": sha_m47,
        "original_m48_lock_sha256": sha_m48,
        "publisher_train_annotation_sha256": source_sha,
        "mean_normalized_absolute_count_error_definition":
            "mean_i(abs(pred_i - GT_i) / GT_i), GT_i > 0",
        "off_by_one_fraction_definition":
            "mean_i(1[abs(pred_i - GT_i) <= 1])",
        "exact_count_fraction_definition":
            "mean_i(1[pred_i == GT_i])",
        "original_absolute_count_mae_definition":
            "mean_i(abs(pred_i - GT_i))",
        "conditions": all_methods,
        "paired_video_percentile_bootstrap": extended_ci,
        "bootstrap_repeats": DRAWS,
        "bootstrap_seed": SEED,
        "frozen_rule_changed": False,
        "source_files_name_disjoint_not_subject_verified": True,
        "publisher_test_or_validation_opened": False,
        "original_m47_scores_recomputed_and_matched": True,
        "raw_traces_or_clip_predictions_persisted": False
    }
    parity_report = {
        "protocol": W6_PROTOCOL,
        "source_split": "train",
        "cohort": "frozen_replication_filename_hash_ranks_21_through_40",
        "videos_evaluated": N,
        "original_m46_report_sha256": sha_m46,
        "original_m48_lock_sha256": sha_m48,
        "publisher_train_annotation_sha256": source_sha,
        "conditions": summarize_w6(parity_rows),
        "same_landmarks_and_frame_timestamps_only": True,
        "browser_mediapipe_tasks_vision_was_run": False,
        "python_vs_browser_pose_estimator_parity_verified": False,
        "actual_browser_requestanimationframe_parity_verified": False,
        "browser_camera_accuracy_measured": False,
        "source_test_or_validation_accessed": False,
        "per_video_gt_raw_landmarks_or_event_times_saved": False
    }
    # Validate before writing. A failed audit produces no misleading output.
    if (report["annotated_gt_reps"] !=
        replayed["frozen_source"]["annotated_gt_reps"] or
        any(v["videos"] != N for v in parity_report["conditions"].values())):
        raise ValueError("M53/W6 post-replay aggregate mismatch")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    w6_report_path.parent.mkdir(parents=True, exist_ok=True)
    # Separate Git-ignored read-only summary reports, never overwrite.
    with report_path.open("x", encoding="utf8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    with w6_report_path.open("x", encoding="utf8") as file:
        json.dump(parity_report, file, indent=2, allow_nan=False)
        file.write("\n")
    return report, parity_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--m46-report", type=Path, default=m47.REFERENCE)
    parser.add_argument("--m47-report", type=Path, default=m47.REPORT)
    parser.add_argument("--m48-lock", type=Path,
                        default=Path("data/processed/m48_train_evidence_freeze.json"))
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--w6-report", type=Path, default=W6_REPORT)
    args = parser.parse_args(argv)
    try:
        m53, w6 = run(
            archive=args.source_archive, root=args.root,
            reference_path=args.m46_report, m47_path=args.m47_report,
            lock_path=args.m48_lock, report_path=args.report,
            w6_report_path=args.w6_report
        )
    except (OSError, RuntimeError, ValueError, TypeError,
            KeyError, json.JSONDecodeError) as error:
        print("M53/W6 refused: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({
        "m53_protocol": m53["protocol"],
        "videos_evaluated": m53["videos_evaluated"],
        "annotated_gt_reps": m53["annotated_gt_reps"],
        "M53_conditions": m53["conditions"],
        "W6_conditions": w6["conditions"],
        "official_test_or_validation_accessed": False,
        "reports": [str(args.report), str(args.w6_report)]
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
