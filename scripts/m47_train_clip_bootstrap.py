"""M47: frozen paired clip-bootstrap audit on M46 unused publisher TRAIN videos.

This repeats EXACTLY the rank-21..40 M46 TRAIN source-file cohort,
unchanged M45 relative excursion, M39 low-to-high, original M34, and
M37 one-to-one event matching. It anchors to the owner's existing
M46 aggregate JSON and refuses output if real-video point estimates
disagree. Bootstrap resampling (5000 repeats, seed 2026) is across
VIDEOS, not GT events, and is descriptive given possible repeated
people, sessions and camera views in those 20 TRAIN videos.

No browser model changes, threshold search, validation/TEST access,
video IDs, raw poses, per-video scores or timestamps in final report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from scripts.m37_repcount_tar import selected_from_tar_archive
from scripts.m37_repcount_video_benchmark import extract_pose_trace
from scripts.m39_repcount_train_baselines import _aggregate
from scripts.m41_repcount_train_cadence import CONDITIONS, SEED
from scripts.m45_train_relative_motion_benchmark import (
    PROTOCOL as M45_PROTOCOL, EXPECTED_RULE,
    compare_node as compare_original_m45, validate_row as validate_original_m45
)

PROTOCOL = "m47-frozen-train-clip-bootstrap-v1"
ROOT = Path("data/external/m47_train_bootstrap")
REPORT = Path("data/processed/m47_train_replication_bootstrap.json")
REFERENCE = Path("data/processed/m46_train_frozen_replication.json")
METHODS = ("low_to_high", "relative_excursion")
BOOTSTRAP_SEED = 2026
BOOTSTRAP_REPEATS = 5000
SOURCE_RANK_OFFSET = 20
EXPECTED_CLIPS = 20


def _per_video(row, condition):
    old = row["original_m44"]["m43"]["m41"][condition]["low_to_high"]
    new = row["conditions"][condition]["relative_excursion"]
    return {
        "low_to_high": old,
        "relative_excursion": new
    }


def _metric(items, field):
    if not items or field not in ("event_f1", "mean_absolute_count_error",
                                 "false_positive_events_per_video"):
        raise ValueError("Unknown M47 bootstrap metric or empty video cohort")
    n = len(items)
    sums = {}
    for name in METHODS:
        clips = [x[name] for x in items]
        tp = sum(c["event_tp"] for c in clips)
        fp = sum(c["event_fp"] for c in clips)
        fn = sum(c["event_fn"] for c in clips)
        if field == "event_f1":
            # Positive publisher GT cohort: micro F1 = 2TP/(2TP+FP+FN),
            # including zero predictions as F1=0 (not null for this summary).
            denom = 2 * tp + fp + fn
            if denom == 0:
                raise ValueError("M47 bootstrap requires positive ground truth")
            value = 2 * tp / denom
        elif field == "mean_absolute_count_error":
            value = sum(c["absolute_count_error"] for c in clips) / n
        else:
            value = fp / n
        sums[name] = value
    return sums["relative_excursion"] - sums["low_to_high"]


def _percentile(sorted_values, fraction):
    if not sorted_values or not 0 <= fraction <= 1:
        raise ValueError("Invalid bootstrap percentile")
    q = fraction * (len(sorted_values) - 1)
    low = int(q)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (q - low) * (
        sorted_values[high] - sorted_values[low]
    )


def paired_video_bootstrap(items, *, repeats=BOOTSTRAP_REPEATS,
                           seed=BOOTSTRAP_SEED):
    """Deterministic 95% percentile intervals; unit = entire source VIDEO.

    Returns no per-video values or samples, and never resamples GT events
    individually because those observations are temporally clustered.
    """
    if (not isinstance(items, list) or len(items) < 2 or
        type(repeats) is not int or not 1 <= repeats <= 100000 or
        type(seed) is not int or not 0 <= seed <= 2**32 - 1):
        raise ValueError("M47 needs >=2 paired videos and valid bootstrap settings")
    n = len(items)
    for clip in items:
        if not isinstance(clip, dict) or set(clip) != set(METHODS):
            raise ValueError("M47 requires both frozen methods on each video")
        ground_truth = None
        for name in METHODS:
            row = clip[name]
            if not isinstance(row, dict):
                raise ValueError("M47 invalid per-clip method stats")
            tp, fp, fn, gt = (row.get(k) for k in (
                "event_tp", "event_fp", "event_fn", "annotated_count"
            ))
            if not all(type(x) is int and x >= 0 for x in (tp, fp, fn, gt)):
                raise ValueError("M47 invalid event totals")
            if tp + fn != gt:
                raise ValueError("M47 source GT/event totals mismatch")
            if row.get("predicted_count") != tp + fp or row.get(
                "absolute_count_error"
            ) != abs(gt - (tp + fp)):
                raise ValueError("M47 source clip counting errors mismatch")
            if ground_truth is None:
                ground_truth = gt
            elif ground_truth != gt:
                raise ValueError("M47 cannot pair methods with different GT")
        if ground_truth <= 0:
            raise ValueError("M47 requires positive-GT TRAIN clips only")
    rng = random.Random(seed)
    fields = (
        "event_f1",
        "mean_absolute_count_error",
        "false_positive_events_per_video"
    )
    deltas = {field: [] for field in fields}
    for _ in range(repeats):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        for field in fields:
            deltas[field].append(_metric(sample, field))
    output = {}
    for field in fields:
        draw = sorted(deltas[field])
        output[field] = {
            "observed_relative_minus_low_to_high": _metric(items, field),
            "percentile_95_interval": [
                _percentile(draw, .025), _percentile(draw, .975)
            ],
            "bootstrap_draws_strictly_above_zero": sum(v > 0 for v in draw),
            "bootstrap_draws_equal_to_zero": sum(v == 0 for v in draw),
            "bootstrap_draws_strictly_below_zero": sum(v < 0 for v in draw)
        }
    return output


def validate_reference(reference, *, annotation_sha, annotated_reps,
                       observed_methods, videos):
    if not isinstance(reference, dict):
        raise ValueError("M47 needs owner's original M46 report object")
    selection = reference.get("source_selection")
    if (reference.get("protocol") != M45_PROTOCOL or
        reference.get("source_split") != "train" or
        not isinstance(selection, dict) or
        selection.get("source_hash_rank_offset") != SOURCE_RANK_OFFSET or
        selection.get("policy") != "stable_hash" or
        selection.get("seed") != SEED or
        selection.get("limit") != EXPECTED_CLIPS or
        selection.get("source_clip_filename_disjoint_from_m40_m45") is not True or
        selection.get("subject_or_view_disjointness_verified") is not False or
        reference.get("relative_excursion_rule") != EXPECTED_RULE or
        reference.get("publisher_train_annotation_sha256") != annotation_sha or
        reference.get("annotated_reps") != annotated_reps or
        reference.get("videos_evaluated") != videos or
        reference.get("videos_excluded") != 0):
        raise ValueError(
            "M47 frozen M46 reference source, rank, protocol or ground truth mismatch"
        )
    conditions = reference.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(CONDITIONS):
        raise ValueError("M47 reference requires both frozen M46 cadences")
    for condition in CONDITIONS:
        actual = observed_methods[condition]
        persisted = conditions[condition].get("methods", {})
        for method_name, row in actual.items():
            ref = persisted.get(method_name)
            if not isinstance(ref, dict):
                raise ValueError("M47 original M46 reference method missing")
            for field in (
                "videos", "annotated_reps", "predicted_reps",
                "event_tp", "event_fp", "event_fn"
            ):
                if type(ref.get(field)) is not int or ref[field] != row[field]:
                    raise ValueError(
                        "M47 reread point estimates differ from owner M46 " +
                        condition + " " + method_name + " " + field
                    )
            for field in (
                "event_f1", "mean_absolute_count_error",
                "mean_signed_count_error"
            ):
                left, right = ref.get(field), row[field]
                if ((left is None) != (right is None)) or (
                    left is not None and (
                        type(left) not in (float, int) or
                        not math.isfinite(left) or
                        not math.isclose(left, right, rel_tol=0, abs_tol=1e-10)
                    )
                ):
                    raise ValueError(
                        "M47 point estimates differ from owner M46 " +
                        condition + " " + method_name + " " + field
                    )


def run(*, archive: Path, root: Path, report_path: Path,
        reference_path: Path, select_fn=None, extract_fn=None,
        compare_fn=None, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED):
    if report_path.exists():
        raise FileExistsError("M47 report already exists; refusing overwrite")
    if not archive.is_file() or not archive.name.lower().endswith((
        ".tar.gz", ".tgz"
    )):
        raise ValueError("M47 requires owner's original authorized TAR.GZ archive")
    if not reference_path.is_file():
        raise FileNotFoundError("M47 requires existing owner's M46 benchmark report")
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    if not isinstance(reference, dict):
        raise ValueError("M47 reference report must be JSON object")
    root.mkdir(parents=True, exist_ok=True)
    selector = select_fn or selected_from_tar_archive
    extractor = extract_fn or extract_pose_trace
    comparator = compare_fn or compare_original_m45
    clips, annotation_sha = selector(
        archive, root, split="train", limit=EXPECTED_CLIPS,
        selection_policy="stable_hash", selection_seed=SEED,
        selection_offset=SOURCE_RANK_OFFSET, require_explicit_split=True
    )
    if len(clips) != EXPECTED_CLIPS:
        raise ValueError("M47 requires full original M46 20-video unused TRAIN cohort")
    if (not isinstance(annotation_sha, str) or len(annotation_sha) != 64 or
        any(c not in "0123456789abcdef" for c in annotation_sha)):
        raise ValueError("M47 publisher TRAIN CSV SHA256 absent")
    records = []
    failures = Counter()
    with tempfile.TemporaryDirectory(dir=root, prefix="m47-private-") as td:
        private = Path(td)
        for i, clip in enumerate(clips):
            try:
                trace = extractor(clip["video_path"], clip, sample_ms=200)
                inp = private / f"pose_{i}.json"
                out = private / f"result_{i}.json"
                with inp.open("x", encoding="utf8") as f:
                    json.dump(trace, f, allow_nan=False)
                evaluated = comparator(inp, out)
                validate_original_m45(evaluated, clip["annotated_count"])
                records.append(evaluated)
            except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
                failures[type(error).__name__] += 1
    if len(records) != EXPECTED_CLIPS:
        raise ValueError(
            "M47 refused uncertainty report: not all 20 frozen TRAIN "
            "clips evaluated (errors by class: " + repr(dict(failures)) + ")"
        )

    observed = {}
    for condition in CONDITIONS:
        items = [_per_video(row, condition) for row in records]
        observed[condition] = {
            method: _aggregate(items, method) for method in METHODS
        }
    gt = sum(row["annotated_rep_count"] for row in records)
    validate_reference(
        reference, annotation_sha=annotation_sha, annotated_reps=gt,
        observed_methods=observed, videos=EXPECTED_CLIPS
    )
    uncertainty = {}
    for condition in CONDITIONS:
        items = [_per_video(row, condition) for row in records]
        uncertainty[condition] = paired_video_bootstrap(
            items, repeats=repeats, seed=seed
        )

    result = {
        "protocol": PROTOCOL,
        "source_split": "train",
        "study_type": "conditional_clip_bootstrap_frozen_M46_train_replication",
        "publisher_train_annotation_sha256": annotation_sha,
        "original_m46_report_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "frozen_source": {
            "selection_policy": "stable_hash",
            "selection_seed": SEED,
            "hash_rank_offset": SOURCE_RANK_OFFSET,
            "videos_evaluated": EXPECTED_CLIPS,
            "videos_excluded": 0,
            "annotated_gt_reps": gt,
            "source_filenames_disjoint_from_original_first_20": True,
            "subject_and_session_disjointness_verified": False
        },
        "rule_unchanged": EXPECTED_RULE,
        "analysis_settings": {
            "paired_cluster_unit": "source_video",
            "bootstrap_method": "percentile",
            "quantiles": [0.025, 0.975],
            "draws": repeats, "seed": seed,
            "difference_orientation": "relative_excursion_minus_low_to_high",
            "two_sided_test_or_p_value_performed": False
        },
        "recomputed_frozen_m46_metrics": observed,
        "paired_clip_bootstrap": uncertainty,
        "limitations": [
            "Bootstrap resamples only 20 publisher TRAIN videos. It does not establish population-level or held-out TEST generalization.",
            "TRAIN source filenames are distinct from original 20, but participants/sessions/views may overlap; 95% intervals may understate dependence.",
            "Candidate M45 and all comparison rules were frozen BEFORE this second TRAIN cohort; any further modifications invalidate its prospective status.",
            "These are percentile clip-resampling intervals, NOT hypothesis-test p-values or an independent participant cluster bootstrap.",
            "Unobserved zero-rep TRAIN controls mean false-alarm specificity remains unmeasured.",
            "Original publisher annotation frame synchronization, dataset origin, pose browser/Python parity and subject disjointness are unverified.",
            "No GT events, raw video, per-clip statistics, poses, timestamps or filenames are persisted.",
            "No original M34 website model altered, no validation or TEST accessed."
        ],
        "deployed_m34_unchanged": True,
        "validation_and_test_untouched": True,
        "raw_or_per_video_data_persisted": False
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, default=REFERENCE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    try:
        result = run(
            archive=args.source_archive, root=args.root,
            report_path=args.report, reference_path=args.reference_report
        )
    except (OSError, RuntimeError, ValueError, TypeError,
            KeyError, json.JSONDecodeError) as error:
        print("M47 stopped: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
