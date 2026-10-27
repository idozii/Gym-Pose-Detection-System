"""M41 paired sampling-density audit on publisher TRAIN only.

This is an EXPLORATORY observation-cadence sensitivity study, not
new-model tuning or a clean comparison of separate MediaPipe tracker
sampling frequencies. ONE dense (~200ms) Python MediaPipe track per
TRAIN video is compared with its every-second-frame subsample.

M34 retains the actual website 400ms internal gate. The M39 low-to-high
detector processes every given tick. Original thresholds, matching,
TRAIN cohort rule and source event annotations remain unchanged.
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
from scripts.m39_repcount_train_baselines import _aggregate, _validate_paired, LABELS

PROTOCOL = "m41-train-cadence-sensitivity-v1"
BASELINE_PROTOCOL = "m39-train-only-temporal-return-v1"
ROOT = Path("data/external/m41_train_cadence")
REPORT = Path("data/processed/m41_train_cadence_report.json")
SCRIPT = Path(__file__).resolve().parents[1] / "web/scripts/m41_compare_train_cadence.js"
CONDITIONS = ("dense_200ms", "subsample_400ms")
SEED = 2026


def compare_node(trace_path: Path, output_path: Path) -> dict:
    try:
        proc = subprocess.run(
            ["node", str(SCRIPT), str(trace_path), str(output_path)],
            capture_output=True, text=True, check=False, timeout=90
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("M41 requires Node 22 and a working paired evaluator") from error
    if proc.returncode:
        raise RuntimeError("M41 Node evaluator failed")
    record = json.loads(output_path.read_text(encoding="utf-8"))
    if record.get("protocol") != PROTOCOL or record.get("source_split") != "train":
        raise ValueError("M41 Node comparison protocol mismatch")
    return record


def _validate(record: dict, annotated_count: int) -> None:
    if not isinstance(record, dict) or record.get("protocol") != PROTOCOL:
        raise ValueError("M41 paired comparison protocol mismatch")
    if record.get("source_split") != "train" or (
            type(record.get("annotated_rep_count")) is not int or
            record["annotated_rep_count"] != annotated_count):
        raise ValueError("M41 only accepts matching TRAIN source counts")
    dense = record.get("dense_200ms")
    sparse = record.get("subsample_400ms")
    if not isinstance(dense, dict) or not isinstance(sparse, dict):
        raise ValueError("M41 missing paired cadence conditions")
    for condition in CONDITIONS:
        row = record[condition]
        if row.get("protocol") != BASELINE_PROTOCOL:
            raise ValueError("M41 condition does not use frozen M39 methods")
        _validate_paired(row, annotated_count)
    n_dense = dense.get("input_frame_samples")
    n_sparse = sparse.get("input_frame_samples")
    if (type(n_dense) is not int or type(n_sparse) is not int or
        n_sparse < 8 or n_dense < 16 or n_sparse != (n_dense + 1) // 2):
        raise ValueError("M41 paired sample lengths not consistent")
    for condition in CONDITIONS:
        row = record[condition]
        if (type(row.get("pose_sample_coverage")) not in (int, float) or
            not math.isfinite(row["pose_sample_coverage"]) or
            not 0 <= row["pose_sample_coverage"] <= 1):
            raise ValueError("M41 invalid pose sample coverage")
    design = record.get("design")
    if (not isinstance(design, dict) or design.get("nominal_dense_ms") != 200 or
        design.get("downsample_factor") != 2 or
        design.get("m34_internal_sample_gate_unchanged_ms") != 400 or
        design.get("shared_python_pose_tracking_context") is not True or
        design.get("exact_independent_400ms_pose_estimator_parity") is not False):
        raise ValueError("M41 frozen comparison design mismatch")


def _paired_error_counts(rows: list[dict], name: str) -> dict:
    """Count per-video lower/tied/higher absolute errors without IDs."""
    improved = tied = worsened = 0
    for row in rows:
        dense = row["dense_200ms"][name]["absolute_count_error"]
        sparse = row["subsample_400ms"][name]["absolute_count_error"]
        if dense < sparse:
            improved += 1
        elif dense == sparse:
            tied += 1
        else:
            worsened += 1
    return {"denser_better": improved, "tied": tied, "denser_worse": worsened}


def run(*, archive: Path, root: Path, report_path: Path, limit: int = 20,
        select_fn=None, extract_fn=None, compare_fn=None) -> dict:
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("M41 expects 1..20 fixed hash TRAIN videos")
    if report_path.exists():
        raise FileExistsError("M41 report exists; refusing overwrite")
    if not archive.is_file() or not archive.name.lower().endswith((".tar.gz", ".tgz")):
        raise ValueError("M41 requires original local author TAR.GZ archive")
    selector = select_fn or selected_from_tar_archive
    extractor = extract_fn or extract_pose_trace
    comparator = compare_fn or compare_node
    root.mkdir(parents=True, exist_ok=True)
    cohort, train_sha = selector(
        archive, root, split="train", limit=limit,
        selection_policy="stable_hash", selection_seed=SEED,
        require_explicit_split=True
    )
    if (not isinstance(train_sha, str) or len(train_sha) != 64 or
        any(ch not in "0123456789abcdef" for ch in train_sha)):
        raise ValueError("M41 requires a SHA256 of the publisher TRAIN CSV")
    observed = []
    exclusions = Counter()
    with tempfile.TemporaryDirectory(dir=root, prefix="m41-traces-") as directory:
        private = Path(directory)
        for i, clip in enumerate(cohort):
            try:
                dense_trace = extractor(clip["video_path"], clip, sample_ms=200)
                inp = private / f"dense_{i}.json"
                out = private / f"paired_{i}.json"
                with inp.open("x", encoding="utf-8") as fd:
                    json.dump(dense_trace, fd, allow_nan=False)
                row = comparator(inp, out)
                _validate(row, clip["annotated_count"])
                observed.append(row)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
                # Privacy: aggregated failure type only, no video paths.
                exclusions[type(error).__name__] += 1
    if not observed:
        raise ValueError("No real TRAIN videos evaluated; refusing accuracy report")
    n = len(observed)
    cohort_reps = sum(x["annotated_rep_count"] for x in observed)
    reports = {}
    for condition in CONDITIONS:
        entries = [item[condition] for item in observed]
        reports[condition] = {
            "input_frame_samples_total": sum(x["input_frame_samples"] for x in entries),
            "mean_pose_sample_coverage": sum(
                x["pose_sample_coverage"] for x in entries
            ) / n,
            "methods": {name: _aggregate(entries, name) for name in LABELS}
        }
    for name in LABELS:
        reports["dense_200ms"]["methods"][name]["paired_count_error"] = (
            _paired_error_counts(observed, name)
        )
    result = {
        "protocol": PROTOCOL,
        "study_type": "exploratory_train_only_observation_cadence_sensitivity",
        "source_mode": "owner_local_author_archive_provenance_unverified",
        "source_split": "train",
        "publisher_train_annotation_sha256": train_sha,
        "source_selection": {
            "policy": "stable_hash", "seed": SEED,
            "requested_limit": limit, "exact_original_train_csv_only": True
        },
        "videos_selected": len(cohort),
        "videos_evaluated": n,
        "videos_excluded": len(cohort) - n,
        "exclusion_reasons": dict(exclusions),
        "annotated_reps": cohort_reps,
        "conditions": reports,
        "interpretation_limits": [
            "Publisher TRAIN only, not independent model validation or TEST.",
            "200ms poses and every-other subsample share the SAME dense MediaPipe Python tracking context.",
            "M34 internal 400ms gate remains unchanged even for 200ms input.",
            "Finer pose sampling may change tracking and computation; no camera/browser parity established.",
            "Same fixed M39 low-to-high rule, M34 and one-to-one event matching in both conditions.",
            "Clips are fixed hash-selected, not demonstrably participant- or view-independent.",
            "No claim of physical squat form correctness, injury safety or held-out model performance."
        ],
        "original_m34_unchanged": True,
        "no_valid_or_test_access": True,
        "raw_frames_or_landmarks_persisted_to_report": False
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as fd:
        json.dump(result, fd, indent=2, allow_nan=False)
        fd.write("\n")
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--max-videos", type=int, default=20)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--report", type=Path, default=REPORT)
    args = p.parse_args(argv)
    try:
        result = run(archive=args.source_archive, root=args.root,
                     report_path=args.report, limit=args.max_videos)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError,
            json.JSONDecodeError) as error:
        print("M41 stopped: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
