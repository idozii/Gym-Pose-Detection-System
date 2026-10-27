"""M39: development-only real Squat benchmark for 3 fixed counting controls.

Uses publisher RepCount-A TRAIN CSV + exact train/video paths within the
owner's local TAR.GZ. Never accesses publisher valid/test clips, new user
recordings, or personal data. Applies original M34 and a separate
predeclared low-to-high rule to IDENTICAL pose traces. Not a trained model.

Only aggregate statistics persist; raw landmark traces are temporary.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.m37_repcount_tar import selected_from_tar_archive
from scripts.m37_repcount_video_benchmark import extract_pose_trace

PROTOCOL = "m39-train-only-temporal-return-v1"
LABELS = ("always_zero", "m34", "low_to_high")
DEFAULT_ROOT = Path("data/external/m39_train_only")
DEFAULT_REPORT = Path("data/processed/m39_train_only_baseline_results.json")
COMPARE_JS = Path(__file__).resolve().parents[1] / "web/scripts/m39_compare_train_trace.js"


def compare_node(trace_file: Path, output_file: Path) -> dict:
    try:
        process = subprocess.run(
            ["node", str(COMPARE_JS), str(trace_file), str(output_file)],
            capture_output=True, text=True, check=False, timeout=90
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("M39 needs Node 22 and a working JS evaluator") from exc
    if process.returncode != 0:
        raise RuntimeError("M39 Node replay failed; inspect local environment")
    report = json.loads(output_file.read_text(encoding="utf-8"))
    if report.get("protocol") != PROTOCOL or report.get("source_split") != "train":
        raise ValueError("Unexpected M39 training replay protocol")
    return report


def _nonnegative_int(value, label):
    if type(value) is not int or value < 0:
        raise ValueError("Invalid M39 nonnegative integer: " + label)
    return value


def _validate_paired(source: dict, annotated_count: int) -> None:
    if source.get("protocol") != PROTOCOL or source.get("source_split") != "train":
        raise ValueError("M39 only accepts its frozen TRAIN replay")
    if _nonnegative_int(source.get("annotation_rep_count"), "source count") != annotated_count:
        raise ValueError("M39 source annotation count disagrees with video")
    for name in LABELS:
        row = source.get(name)
        if not isinstance(row, dict):
            raise ValueError("M39 missing method: " + name)
        pred = _nonnegative_int(row.get("predicted_count"), name + " predictions")
        tp = _nonnegative_int(row.get("event_tp"), name + " tp")
        fp = _nonnegative_int(row.get("event_fp"), name + " fp")
        fn = _nonnegative_int(row.get("event_fn"), name + " fn")
        if (row.get("annotated_count") != annotated_count or
            tp + fn != annotated_count or tp + fp != pred or
            row.get("absolute_count_error") != abs(pred - annotated_count) or
            row.get("signed_count_error") != pred - annotated_count):
            raise ValueError("M39 mismatch in " + name + " count/event totals")
    zero = source["always_zero"]
    if zero["predicted_count"] != 0 or zero["event_tp"] or zero["event_fp"]:
        raise ValueError("Always-zero control made positive predictions")
    coverage = source.get("pose_sample_coverage")
    if type(coverage) not in (float, int) or not math.isfinite(coverage) or not 0 <= coverage <= 1:
        raise ValueError("Invalid M39 per-video pose coverage")


def _aggregate(entries: list[dict], name: str) -> dict:
    n = len(entries)
    rows = [e[name] for e in entries]
    tp = sum(r["event_tp"] for r in rows)
    fp = sum(r["event_fp"] for r in rows)
    fn = sum(r["event_fn"] for r in rows)
    pred = sum(r["predicted_count"] for r in rows)
    gt = sum(r["annotated_count"] for r in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall
          else 0.0 if precision is not None and recall is not None else None)
    return {
        "videos": n,
        "annotated_reps": gt,
        "predicted_reps": pred,
        "event_tp": tp, "event_fp": fp, "event_fn": fn,
        "event_precision": precision, "event_recall": recall,
        "event_f1": f1,
        "mean_absolute_count_error": sum(r["absolute_count_error"] for r in rows) / n,
        "mean_signed_count_error": sum(r["signed_count_error"] for r in rows) / n,
        "exact_count_fraction": sum(r["absolute_count_error"] == 0 for r in rows) / n,
        "within_one_fraction": sum(r["absolute_count_error"] <= 1 for r in rows) / n,
        "zero_rep_negative_control_videos": sum(r["annotated_count"] == 0 for r in rows),
        "zero_rep_negative_controls_with_false_cycles": sum(
            r["annotated_count"] == 0 and r["predicted_count"] > 0 for r in rows
        )
    }


def run(*, archive: Path, root: Path, report_path: Path, limit: int = 5,
        select_fn=None, extract_fn=None, compare_fn=None,
        selection_policy: str = "archive_order", selection_seed: int = 2026) -> dict:
    if report_path.exists():
        raise FileExistsError("M39 output exists; refusing overwrite")
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("M39 permits only 1..20 deterministic publisher TRAIN videos")
    if not archive.is_file() or not archive.name.lower().endswith((".tar.gz", ".tgz")):
        raise ValueError("M39 requires existing authorized RepCountA.tar.gz")
    if selection_policy not in ("archive_order", "stable_hash"):
        raise ValueError("Unknown M39 selection policy")
    if type(selection_seed) is not int or not 0 <= selection_seed <= 2**32 - 1:
        raise ValueError("M39 selection seed must be a nonnegative 32-bit integer")
    root.mkdir(parents=True, exist_ok=True)
    select = select_fn or selected_from_tar_archive
    options = {"split": "train", "limit": limit, "require_explicit_split": True}
    if selection_policy == "stable_hash":
        options.update(selection_policy=selection_policy, selection_seed=selection_seed)
    cohort, source_hash = select(archive, root, **options)
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("M39 requires original TRAIN annotation SHA256")
    collected = []
    excluded = Counter()
    extractor = extract_fn or extract_pose_trace
    comparator = compare_fn or compare_node
    with tempfile.TemporaryDirectory(prefix="m39-trace-", dir=root) as tempdir:
        private = Path(tempdir)
        for i, entry in enumerate(cohort):
            try:
                trace = extractor(entry["video_path"], entry)
                trace_path = private / f"trace_{i}.json"
                comparison_path = private / f"comparison_{i}.json"
                with trace_path.open("x", encoding="utf-8") as f:
                    json.dump(trace, f, allow_nan=False)
                comparison = comparator(trace_path, comparison_path)
                _validate_paired(comparison, entry["annotated_count"])
                collected.append(comparison)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
                # No clip identifiers/frames/poses in aggregate exclusion reasons.
                excluded[type(error).__name__] += 1
    if not collected:
        raise ValueError("No publisher TRAIN Squat clips evaluated; no M39 accuracy report")
    total_annotated = sum(x["annotation_rep_count"] for x in collected)
    report = {
        "protocol": PROTOCOL,
        "study_type": "development_only_exploratory_method_comparison",
        "source_mode": "user_provided_author_archive_provenance_unverified",
        "source_split": "train",
        "source_annotation_sha256": source_hash,
        "selection_policy": selection_policy,
        "selection_seed": selection_seed if selection_policy == "stable_hash" else None,
        "requested_max_videos": limit,
        "candidate_videos": len(cohort),
        "evaluated_videos": len(collected),
        "excluded_videos": len(cohort) - len(collected),
        "exclusion_reasons": dict(excluded),
        "annotated_reps": total_annotated,
        "pose_sample_coverage_mean": sum(
            x["pose_sample_coverage"] for x in collected
        ) / len(collected),
        "frozen_methods": {
            "always_zero": "Always predict 0; precision/F1 null if no predictions",
            "m34": "Unmodified original browser M34 RepCycleCounter",
            "low_to_high": (
                "One observed knee <=120deg then >=155deg on same tracked "
                "side, >=35deg excursion, 400-12000ms, tracking gap <=1250ms"
            )
        },
        "results": {name: _aggregate(collected, name) for name in LABELS},
        "inference_limits": [
            "TRAIN cohort only: neither publisher validation nor TEST touched.",
            "The low-to-high rule is an untrained developmental baseline.",
            "Video selection uses a deterministic development-only policy; small N is not population inference.",
            "Hash-selected filenames do NOT establish participant-disjoint sampling.",
            "Publisher annotation/video synchronization and browser/Python estimator parity unverified.",
            "No exercise form correctness or injury-safety classification.",
            "Original pose traces and video names never written to the aggregate JSON."
        ],
        "train_data_only": True,
        "validation_or_test_accessed": False,
        "original_m34_modified": False
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--max-videos", type=int, default=5)
    parser.add_argument("--selection-policy", choices=("archive_order", "stable_hash"),
                        default="archive_order")
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        report = run(archive=args.source_archive, root=args.root,
                     report_path=args.report, limit=args.max_videos,
                     selection_policy=args.selection_policy,
                     selection_seed=args.selection_seed)
    except (OSError, RuntimeError, ValueError, TypeError,
            json.JSONDecodeError) as error:
        print(f"M39 stopped: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
