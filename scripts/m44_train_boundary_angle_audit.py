"""M44: publisher TRAIN-only noncausal angular/boundary diagnostics.

Replicates M43/M41 frozen evaluation on 20 source TRAIN clips with
200ms-target pose observations. The new diagnostics examine angles
inside half-open publisher GT windows and in fixed +/-750ms context.
No detector thresholds change and no source validation/TEST are used.

Only aggregate statistics persist. Video/poses/events stay ephemeral.
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
from scripts.m39_repcount_train_baselines import _aggregate
from scripts.m41_repcount_train_cadence import (
    _validate as validate_m41, CONDITIONS, LABELS, SEED
)
from scripts.m43_temporal_error_schema import (
    validate_m43_diagnostics, aggregate_diagnostics
)
from scripts.m44_boundary_angle_schema import (
    PROTOCOL, CONTEXT_MS, validate_against_m43, aggregate
)

ROOT = Path("data/external/m44_train_boundary_audit")
REPORT = Path("data/processed/m44_train_boundary_angle_audit.json")
NODE = Path(__file__).resolve().parents[1] / "web/scripts/m44_train_boundary_audit.js"


def compare_node(input_file: Path, output_file: Path) -> dict:
    try:
        proc = subprocess.run(
            ["node", str(NODE), str(input_file), str(output_file)],
            capture_output=True, text=True, check=False, timeout=90
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("M44 requires working Node 22 replay") from error
    if proc.returncode:
        raise RuntimeError("M44 Node error analysis failed")
    return json.loads(output_file.read_text(encoding="utf8"))


def run(*, archive: Path, root: Path, report_path: Path, limit: int = 20,
        select_fn=None, extract_fn=None, compare_fn=None) -> dict:
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("M44 expects 1..20 fixed SHA256-selected TRAIN clips")
    if report_path.exists():
        raise FileExistsError("M44 output exists; refusing overwrite")
    if not archive.is_file() or not archive.name.lower().endswith((".tar.gz", ".tgz")):
        raise ValueError("M44 requires an existing owner-local RepCountA.tar.gz")
    selector = select_fn or selected_from_tar_archive
    extractor = extract_fn or extract_pose_trace
    comparator = compare_fn or compare_node
    root.mkdir(parents=True, exist_ok=True)
    selected, source_hash = selector(
        archive, root, split="train", limit=limit,
        require_explicit_split=True,
        selection_policy="stable_hash", selection_seed=SEED
    )
    if (not isinstance(source_hash, str) or len(source_hash) != 64 or
        any(c not in "0123456789abcdef" for c in source_hash)):
        raise ValueError("M44 needs original publisher TRAIN annotation SHA256")
    observed = []
    excluded = Counter()
    with tempfile.TemporaryDirectory(dir=root, prefix="m44-private-") as folder:
        private = Path(folder)
        for i, video in enumerate(selected):
            try:
                trace = extractor(video["video_path"], video, sample_ms=200)
                inp = private / f"sample_{i}.json"
                out = private / f"analysis_{i}.json"
                with inp.open("x", encoding="utf8") as f:
                    json.dump(trace, f, allow_nan=False)
                row = comparator(inp, out)
                if (not isinstance(row, dict) or
                    row.get("protocol") != PROTOCOL or
                    row.get("source_split") != "train" or
                    row.get("annotated_rep_count") != video["annotated_count"] or
                    row.get("context_ms") != CONTEXT_MS):
                    raise ValueError("M44 per-video TRAIN/source protocol mismatch")
                m43 = row["m43"]
                if (m43.get("protocol") != "m43-train-temporal-error-audit-v1" or
                    m43.get("source_split") != "train" or
                    m43.get("annotated_rep_count") != video["annotated_count"]):
                    raise ValueError("M44 requires exact frozen M43 report")
                validate_m41(m43["m41"], video["annotated_count"])
                validate_m43_diagnostics(m43["diagnostics"], m43["m41"])
                validate_against_m43(row["conditions"], m43["diagnostics"])
                observed.append(row)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
                excluded[type(error).__name__] += 1
    if not observed:
        raise ValueError("No publisher TRAIN videos evaluated; no M44 accuracy report")
    n = len(observed)
    angular = aggregate([row["conditions"] for row in observed])
    m43 = aggregate_diagnostics([row["m43"]["diagnostics"] for row in observed])
    comparisons = {}
    for cond in CONDITIONS:
        baselines = [row["m43"]["m41"][cond] for row in observed]
        comparisons[cond] = {
            "input_frame_samples_total": sum(
                row["input_frame_samples"] for row in baselines
            ),
            "methods": {
                name: _aggregate(baselines, name) for name in LABELS
            }
        }
        e = comparisons[cond]["methods"]["low_to_high"]
        m = m43["conditions"][cond]
        if (m["event_tp"] != e["event_tp"] or
            m["event_fp"] != e["event_fp"] or
            m["event_fn"] != e["event_fn"]):
            raise ValueError("M44 M41/M43 cohort event totals disagree")
        if angular[cond]["unmatched_gt_no_low_inside"] != m[
            "unmatched_gt_window_observations"
        ]["usable_but_no_low_in_gt_window"]:
            raise ValueError("M44 cohort no-low bins disagree with M43")
        if sum(angular[cond]["unmatched_prediction_location"].values()) != e["event_fp"]:
            raise ValueError("M44 cohort FP bins disagree with original matcher")
    result = {
        "protocol": PROTOCOL,
        "study_type": "post_hoc_train_only_noncausal_boundary_angle_diagnostics",
        "source_split": "train",
        "source_mode": "owner_local_author_archive_provenance_unverified",
        "publisher_train_annotation_sha256": source_hash,
        "selection": {"policy": "stable_hash", "seed": SEED,
                      "limit": limit, "target_sample_ms": 200,
                      "same_dense_mediapipe_history_for_subsample": True},
        "videos_selected": len(selected),
        "videos_evaluated": n,
        "videos_excluded": len(selected) - n,
        "excluded_reasons": dict(excluded),
        "annotated_reps": sum(row["annotated_rep_count"] for row in observed),
        "context_window_ms": CONTEXT_MS,
        "frozen_m41_metrics": comparisons,
        "frozen_m43_diagnostics": m43,
        "m44_boundary_diagnostics": angular,
        "limitations": [
            "TRAIN-only post-hoc observations, NOT held-out TEST or causal explanations.",
            "Unmatched GT windows with no low inside may have low poses before or after source boundaries.",
            "The 750ms before/after windows can overlap neighboring publisher repetitions.",
            "No observed low in a short window does not prove no deep squat or incorrect motion.",
            "Angle minima are 2D camera projections from Python MediaPipe tracking, not true joint angles.",
            "200ms and subsample 400ms share the same MediaPipe tracking history.",
            "Publisher source frame synchronization, camera view and subjects unverified.",
            "M39 thresholds, original M34 and M37 one-to-one matching stay unchanged.",
            "No publisher TRAIN Squat zero-rep control exists, so specificity remains unmeasured.",
            "No video names, source poses, timestamps or per-event records are persisted."
        ],
        "validation_or_test_accessed": False,
        "original_detectors_modified": False,
        "raw_source_data_persisted": False
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
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
        print("M44 stopped: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
