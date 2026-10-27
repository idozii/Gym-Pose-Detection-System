"""M45: exploratory publisher TRAIN relative-vs-absolute Squat counting trial.

Hypothesis was selected AFTER owner-visible M43/M44 results; even a large
improvement on the same 20 clips is not independent model validation.

Same publisher TRAIN stable-hash seed, same dense 200ms MediaPipe poses,
same GT, original M37 matcher. M34/M39/M41/M43/M44 unchanged.
Only one predeclared nondeployed relative-excursion detector is evaluated.
All video traces and per-video event records are temporary local data.
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
from scripts.m41_repcount_train_cadence import (
    _validate as validate_m41, CONDITIONS, SEED
)
from scripts.m43_temporal_error_schema import (
    validate_m43_diagnostics, aggregate_diagnostics
)
from scripts.m44_boundary_angle_schema import (
    PROTOCOL as M44_PROTOCOL,
    validate_against_m43, aggregate as aggregate_m44
)

PROTOCOL = "m45-train-relative-excursion-comparator-v1"
EXPECTED_RULE = {
    "highDeg": 155,
    "minDropDeg": 35,
    "minRiseDeg": 35,
    "minReturnMs": 400,
    "maxReturnMs": 12000,
    "maxTrackingGapMs": 1250
}
ROOT = Path("data/external/m45_train_relative_motion")
REPORT = Path("data/processed/m45_train_relative_motion_report.json")
NODE = Path(__file__).resolve().parents[1] / "web/scripts/m45_compare_train_relative.js"
PAIR_KEYS = (
    "both_matched", "new_only_matched",
    "original_only_matched", "neither_matched"
)
FP_KEYS = (
    "unmatched_inside_any_gt_interval",
    "unmatched_outside_all_gt_intervals"
)
RESET_KEYS = ("side", "gap", "timeout")


def compare_node(trace_path: Path, result_path: Path) -> dict:
    try:
        proc = subprocess.run(
            ["node", str(NODE), str(trace_path), str(result_path)],
            capture_output=True, text=True, check=False, timeout=90
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("M45 requires Node 22 and a working M45 JS replay") from error
    if proc.returncode:
        raise RuntimeError("M45 Node scorer failed")
    return json.loads(result_path.read_text(encoding="utf8"))


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("M45 requires nonnegative event count " + name)
    return value


def _counters(data, names, label):
    if not isinstance(data, dict) or set(data) != set(names):
        raise ValueError("M45 invalid " + label + " schema")
    return {name: _integer(data[name], label + ":" + name) for name in names}


def _validate_candidate(method, n):
    if not isinstance(method, dict):
        raise ValueError("M45 candidate method missing")
    pred = _integer(method.get("predicted_count"), "predicted")
    tp = _integer(method.get("event_tp"), "tp")
    fp = _integer(method.get("event_fp"), "fp")
    fn = _integer(method.get("event_fn"), "fn")
    if (method.get("annotated_count") != n or tp + fn != n or
        tp + fp != pred or
        method.get("absolute_count_error") != abs(pred - n) or
        method.get("signed_count_error") != pred - n):
        raise ValueError("M45 candidate matched/count totals disagree")
    precision = tp / pred if pred else None
    recall = tp / n if n else None
    f1 = None if precision is None or recall is None else (
        0.0 if precision + recall == 0 else
        2 * precision * recall / (precision + recall)
    )
    for field, expected in (
        ("event_precision", precision), ("event_recall", recall),
        ("event_f1", f1)
    ):
        actual = method.get(field)
        if (actual is None) != (expected is None) or (
            expected is not None and (
                type(actual) not in (int, float) or
                not math.isfinite(actual) or
                not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
            )
        ):
            raise ValueError("M45 candidate " + field + " metric disagrees")


def validate_row(row: dict, n: int) -> None:
    if (not isinstance(row, dict) or row.get("protocol") != PROTOCOL or
        row.get("source_split") != "train" or
        row.get("annotated_rep_count") != n or
        row.get("rule") != EXPECTED_RULE):
        raise ValueError("M45 frozen protocol, rule or TRAIN GT mismatch")
    original = row.get("original_m44")
    if (not isinstance(original, dict) or
        original.get("protocol") != M44_PROTOCOL or
        original.get("source_split") != "train" or
        original.get("annotated_rep_count") != n or
        original.get("context_ms") != 750):
        raise ValueError("M45 must replay original M44 unmodified")
    m43 = original["m43"]
    if (m43.get("protocol") != "m43-train-temporal-error-audit-v1" or
        m43.get("source_split") != "train" or
        m43.get("annotated_rep_count") != n):
        raise ValueError("M45 original M43 protocol mismatched")
    validate_m41(m43["m41"], n)
    validate_m43_diagnostics(m43["diagnostics"], m43["m41"])
    validate_against_m43(original["conditions"], m43["diagnostics"])
    conditions = row.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(CONDITIONS):
        raise ValueError("M45 expects two source paired-cadence comparisons")
    for cond in CONDITIONS:
        c = conditions[cond]
        if not isinstance(c, dict) or set(c) != {
            "relative_excursion", "paired_gt_vs_original_low_to_high",
            "candidate_fp_location", "pending_resets"
        }:
            raise ValueError("M45 incomplete comparator condition")
        _validate_candidate(c["relative_excursion"], n)
        base = m43["m41"][cond]
        _validate_paired(base, n)
        paired = _counters(c["paired_gt_vs_original_low_to_high"],
                           PAIR_KEYS, "paired GT")
        relative = c["relative_excursion"]
        original_low = base["low_to_high"]
        if (sum(paired.values()) != n or
            paired["both_matched"] + paired["new_only_matched"] !=
            relative["event_tp"] or
            paired["both_matched"] + paired["original_only_matched"] !=
            original_low["event_tp"]):
            raise ValueError("M45 paired matching disagrees with original method")
        fp = _counters(c["candidate_fp_location"], FP_KEYS, "false positives")
        if sum(fp.values()) != relative["event_fp"]:
            raise ValueError("M45 candidate FP location partitions disagree")
        _counters(c["pending_resets"], RESET_KEYS, "pending resets")


def _aggregate_pairwise(records, condition):
    totals = {key: 0 for key in PAIR_KEYS}
    fp = {key: 0 for key in FP_KEYS}
    resets = {key: 0 for key in RESET_KEYS}
    paired_count = {"relative_better": 0, "tied": 0, "relative_worse": 0}
    for row in records:
        candidate = row["conditions"][condition]
        orig = row["original_m44"]["m43"]["m41"][condition]["low_to_high"]
        for key in PAIR_KEYS:
            totals[key] += candidate["paired_gt_vs_original_low_to_high"][key]
        for key in FP_KEYS:
            fp[key] += candidate["candidate_fp_location"][key]
        for key in RESET_KEYS:
            resets[key] += candidate["pending_resets"][key]
        a = candidate["relative_excursion"]["absolute_count_error"]
        b = orig["absolute_count_error"]
        if a < b:
            paired_count["relative_better"] += 1
        elif a == b:
            paired_count["tied"] += 1
        else:
            paired_count["relative_worse"] += 1
    return {
        "paired_gt_vs_original_low_to_high": totals,
        "candidate_fp_location": fp,
        "candidate_pending_resets": resets,
        "paired_video_absolute_count_error": paired_count
    }


def run(*, archive: Path, root: Path, report_path: Path, limit: int = 20,
        select_fn=None, extract_fn=None, compare_fn=None,
        cohort_offset: int = 0) -> dict:
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("M45 requires 1..20 fixed hash publisher TRAIN clips")
    # Exactly one preregistered fresh-TRAIN replication offset is offered.
    # Prevent outcome-guided browsing of arbitrary cohort offsets.
    if type(cohort_offset) is not int or cohort_offset not in (0, 20):
        raise ValueError("M46 only allows frozen hash cohort offsets 0 or 20")
    if report_path.exists():
        raise FileExistsError("M45 output exists; refusing overwrite")
    if not archive.is_file() or not archive.name.lower().endswith((".tar.gz", ".tgz")):
        raise ValueError("M45 needs an existing owner-local RepCountA.tar.gz")
    selector = select_fn or selected_from_tar_archive
    extractor = extract_fn or extract_pose_trace
    comparator = compare_fn or compare_node
    root.mkdir(parents=True, exist_ok=True)
    select_options = {
        "split": "train", "limit": limit, "require_explicit_split": True,
        "selection_policy": "stable_hash", "selection_seed": SEED
    }
    # Preserve exactly the original M45 loader API for offset=0.
    if cohort_offset:
        select_options["selection_offset"] = cohort_offset
    clips, train_sha = selector(archive, root, **select_options)
    if (not isinstance(train_sha, str) or len(train_sha) != 64 or
        any(ch not in "0123456789abcdef" for ch in train_sha)):
        raise ValueError("M45 requires original publisher TRAIN CSV SHA256")
    rows, exclusions = [], Counter()
    with tempfile.TemporaryDirectory(prefix="m45-private-", dir=root) as tmp:
        private = Path(tmp)
        for i, clip in enumerate(clips):
            try:
                trace = extractor(clip["video_path"], clip, sample_ms=200)
                source = private / f"pose_{i}.json"
                result = private / f"scored_{i}.json"
                with source.open("x", encoding="utf8") as fd:
                    json.dump(trace, fd, allow_nan=False)
                row = comparator(source, result)
                validate_row(row, clip["annotated_count"])
                rows.append(row)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
                # Aggregate exception type only, never source path or clip ID.
                exclusions[type(error).__name__] += 1
    if not rows:
        raise ValueError("No TRAIN videos evaluated; M45 cannot produce accuracy")
    outcomes = {}
    for condition in CONDITIONS:
        originals = [row["original_m44"]["m43"]["m41"][condition] for row in rows]
        candidates = [{"relative_excursion": row["conditions"][condition][
            "relative_excursion"]} for row in rows]
        methods = {name: _aggregate(originals, name) for name in LABELS}
        methods["relative_excursion"] = _aggregate(candidates, "relative_excursion")
        pairing = _aggregate_pairwise(rows, condition)
        gt = methods["low_to_high"]["annotated_reps"]
        if (gt != methods["relative_excursion"]["annotated_reps"] or
            sum(pairing["paired_gt_vs_original_low_to_high"].values()) != gt or
            sum(pairing["candidate_fp_location"].values()) !=
            methods["relative_excursion"]["event_fp"]):
            raise ValueError("M45 cohort relative/original event metrics inconsistent")
        if sum(pairing["paired_video_absolute_count_error"].values()) != len(rows):
            raise ValueError("M45 per-video paired error count not exhaustive")
        outcomes[condition] = {
            "input_frame_samples_total": sum(x["input_frame_samples"] for x in originals),
            "methods": methods, "paired_diagnostics": pairing
        }
    m43 = aggregate_diagnostics([row["original_m44"]["m43"]["diagnostics"]
                                 for row in rows])
    original_m44 = aggregate_m44([row["original_m44"]["conditions"] for row in rows])
    for cond in CONDITIONS:
        base = outcomes[cond]["methods"]["low_to_high"]
        old = m43["conditions"][cond]
        if (base["event_tp"] != old["event_tp"] or
            base["event_fp"] != old["event_fp"] or
            base["event_fn"] != old["event_fn"] or
            original_m44[cond]["unmatched_gt_no_low_inside"] != old[
                "unmatched_gt_window_observations"][
                "usable_but_no_low_in_gt_window"]):
            raise ValueError("M45 frozen M43/M44 aggregate reproduction failed")
    report = {
        "protocol": PROTOCOL,
        "study_type": (
            "prespecified_unseen_train_source_file_replication"
            if cohort_offset == 20 else
            "post_hoc_exploratory_train_only_relative_excursion"
        ),
        "source_split": "train",
        "source_mode": "owner_local_author_archive_provenance_unverified",
        "publisher_train_annotation_sha256": train_sha,
        "source_selection": {
            "policy": "stable_hash", "seed": SEED, "limit": limit,
            "source_hash_rank_offset": cohort_offset,
            "source_clip_filename_disjoint_from_m40_m45": cohort_offset == 20,
            "subject_or_view_disjointness_verified": False,
            "pose_observation_target_ms": 200,
            "400ms_subsample_shares_dense_tracker_history": True
        },
        "videos_selected": len(clips), "videos_evaluated": len(rows),
        "videos_excluded": len(clips) - len(rows),
        "exclusion_reasons": dict(exclusions),
        "annotated_reps": sum(row["annotated_rep_count"] for row in rows),
        "relative_excursion_rule": EXPECTED_RULE,
        "conditions": outcomes,
        "original_m43_diagnostics": m43,
        "original_m44_diagnostics": original_m44,
        "interpretation_limits": [
            (
                "The relative rule was frozen after observing the first 20 TRAIN clips; "
                "the new hash-rank-offset-20 source videos were not used to design it. "
                "They remain publisher TRAIN, not TEST; person/view disjointness is unknown."
                if cohort_offset == 20 else
                "The relative algorithm was designed AFTER reviewing these exact TRAIN data; "
                "all scores are exploratory and may be overfit."
            ),
            "Candidate counts are not evidence of valid exercise form, biomechanics or injury safety.",
            "FP rise and per-video worsened counts must be considered alongside recovered TP.",
            "The cohort contains no publisher-labeled zero-rep Squat TRAIN clips. Negative-control specificity is unavailable.",
            "Same dense Python MediaPipe tracking history is reused for every-other 400ms condition.",
            "All source annotation end-tolerance matching is unchanged; publisher boundary synchronization remains unverified.",
            "No validation/TEST media, no detector deployment, and no independent generalization claims."
        ],
        "no_validation_or_test_access": True,
        "deployed_rep_counter_unchanged": True,
        "no_video_names_poses_or_timestamps_persisted": True
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf8") as fd:
        json.dump(report, fd, indent=2, allow_nan=False)
        fd.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--cohort-offset", type=int, choices=(0, 20),
                        default=0,
                        help="0: original M45 TRAIN development set; "
                             "20: M46 frozen next-hash-rank TRAIN replication")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    try:
        output = run(archive=args.source_archive, root=args.root,
                     report_path=args.report, limit=args.max_videos,
                     cohort_offset=args.cohort_offset)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError,
            json.JSONDecodeError) as error:
        print("M45 stopped: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
