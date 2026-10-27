"""M43: publisher TRAIN-only aggregate temporal error analysis of M41.

No detector tuning. Re-extracts the exact fixed-hash TRAIN cohort at
200ms and compares original M41 methods and event matching. The
classification of unmatched publisher half-open intervals is strictly
DESCRIPTIVE, not proof of why a repetition was missed.

Raw poses, video labels, per-event timestamps and per-video diagnostics
exist only in short-lived local temporary files; persist aggregates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.m37_repcount_tar import selected_from_tar_archive
from scripts.m37_repcount_video_benchmark import extract_pose_trace
from scripts.m41_repcount_train_cadence import (
    _validate as validate_m41, _paired_error_counts, CONDITIONS, LABELS,
    SEED
)
from scripts.m39_repcount_train_baselines import _aggregate
from scripts.m43_temporal_error_schema import (
    PROTOCOL, FN_KEYS, FP_KEYS, validate_m43_diagnostics, aggregate_diagnostics
)

ROOT = Path("data/external/m43_train_temporal_errors")
REPORT = Path("data/processed/m43_train_temporal_error_audit.json")
SCRIPT = Path(__file__).resolve().parents[1] / "web/scripts/m43_analyse_train_errors.js"


def compare_node(trace_path: Path, result_path: Path) -> dict:
    try:
        result = subprocess.run(
            ["node", str(SCRIPT), str(trace_path), str(result_path)],
            capture_output=True, text=True, timeout=90, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("M43 requires Node 22 and the M43 JS evaluator") from error
    if result.returncode:
        raise RuntimeError("M43 Node evaluator failed; review local setup")
    return json.loads(result_path.read_text(encoding="utf8"))


def run(*, archive: Path, root: Path, report_path: Path, limit: int = 20,
        select_fn=None, extract_fn=None, compare_fn=None) -> dict:
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("M43 requires 1..20 fixed-hash TRAIN videos")
    if report_path.exists():
        raise FileExistsError("M43 report exists; refusing overwrite")
    if not archive.is_file() or not archive.name.lower().endswith((".tar.gz", ".tgz")):
        raise ValueError("M43 requires existing local RepCountA TAR.GZ")
    selector = select_fn or selected_from_tar_archive
    extractor = extract_fn or extract_pose_trace
    comparator = compare_fn or compare_node
    root.mkdir(parents=True, exist_ok=True)
    cohort, source_hash = selector(
        archive, root, split="train", limit=limit,
        require_explicit_split=True, selection_policy="stable_hash",
        selection_seed=SEED
    )
    if (not isinstance(source_hash, str) or len(source_hash) != 64 or
        any(c not in "0123456789abcdef" for c in source_hash)):
        raise ValueError("M43 expected publisher TRAIN annotation SHA256")
    records = []
    excluded = Counter()
    with tempfile.TemporaryDirectory(prefix="m43-private-", dir=root) as private_dir:
        private = Path(private_dir)
        for i, clip in enumerate(cohort):
            try:
                trace = extractor(clip["video_path"], clip, sample_ms=200)
                pose_path = private / f"trace_{i}.json"
                per_video_path = private / f"diagnostic_{i}.json"
                with pose_path.open("x", encoding="utf8") as stream:
                    json.dump(trace, stream, allow_nan=False)
                row = comparator(pose_path, per_video_path)
                if not isinstance(row, dict) or row.get("protocol") != PROTOCOL or (
                    row.get("source_split") != "train" or
                    row.get("annotated_rep_count") != clip["annotated_count"]
                ):
                    raise ValueError("M43 paired TRAIN cohort/source count mismatch")
                validate_m41(row.get("m41"), clip["annotated_count"])
                validate_m43_diagnostics(row["diagnostics"], row["m41"])
                records.append(row)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
                # Only the exception CLASS survives; names/poses remain private.
                excluded[type(error).__name__] += 1
    if not records:
        raise ValueError("No TRAIN videos evaluated; M43 will not invent accuracy")
    n = len(records)
    comparisons = {}
    for condition in CONDITIONS:
        baseline = [entry["m41"][condition] for entry in records]
        comparisons[condition] = {
            "input_frame_samples_total": sum(x["input_frame_samples"] for x in baseline),
            "mean_pose_sample_coverage": sum(x["pose_sample_coverage"] for x in baseline) / n,
            "methods": {name: _aggregate(baseline, name) for name in LABELS}
        }
        for name in LABELS:
            comparisons[condition]["methods"][name]["paired_count_error"] = (
                _paired_error_counts([x["m41"] for x in records], name)
            )

    diagnostic = aggregate_diagnostics([entry["diagnostics"] for entry in records])
    for condition in CONDITIONS:
        m = comparisons[condition]["methods"]["low_to_high"]
        d = diagnostic["conditions"][condition]
        if (d["event_tp"] != m["event_tp"] or d["event_fp"] != m["event_fp"] or
            d["event_fn"] != m["event_fn"] or
            d["gt_events"] != m["annotated_reps"]):
            raise ValueError("M43 cohort error analysis disagrees with M41 totals")
    result = {
        "protocol": PROTOCOL,
        "study_type": "post_hoc_exploratory_train_temporal_error_analysis",
        "source_split": "train",
        "source_mode": "owner_local_publisher_archive_provenance_unverified",
        "publisher_train_annotation_sha256": source_hash,
        "selection_policy": "stable_hash",
        "selection_seed": SEED,
        "requested_limit": limit,
        "videos_selected": len(cohort),
        "videos_evaluated": n,
        "videos_excluded": len(cohort) - n,
        "exclusion_reasons": dict(excluded),
        "annotated_reps": sum(e["annotated_rep_count"] for e in records),
        "frozen_m41_method_metrics": comparisons,
        "m43_diagnostics": diagnostic,
        "limitations": [
            "TRAIN only. All error buckets are descriptive, NOT causal explanations.",
            "Publisher intervals are half-open and may exclude a low/high endpoint outside the interval.",
            "The one-to-one M37 matcher allows unmatched predictions inside another GT event tolerance.",
            "200ms and every-other 400ms traces share the same dense MediaPipe tracking context.",
            "M34 internal 400ms gate, M39 thresholds, M37 matching and train sampling are unchanged.",
            "No original Squat zero-rep negative controls in the publisher TRAIN CSV (M42 found none).",
            "No genuine TEST accuracy, subject independence, exercise quality or injury-safety claims.",
            "Source videos, landmarks, timestamps and per-video event results are not retained in the report."
        ],
        "validation_or_test_accessed": False,
        "original_m34_modified": False,
        "raw_video_or_landmarks_persisted": False
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    try:
        result = run(archive=args.source_archive, root=args.root,
                     report_path=args.report, limit=args.max_videos)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError,
            json.JSONDecodeError) as error:
        print("M43 stopped: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
