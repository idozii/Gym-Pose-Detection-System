"""M50: audit existing publisher Squat TRAIN GT windows and video timing only.

M50 does NOT reevaluate a rep counter, modify thresholds, open TEST or
validation CSV/video contents, or make subject-independence claims.

Precondition: owner has the original M45 (hash ranks 1-20) and M46
(hash ranks 21-40) extracted TRAIN clips, and valid original M46/47/48
evidence locks. No inference/MediaPipe/GPU run occurs.

The original archive is streamed once to read only TRAIN annotation
CSV; no video member extracted. Locally pre-extracted TRAIN MP4s
are fully decoded for timing consistency, but no frames are retained.
A raw source video byte-for-byte comparison to archive is not done.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

from scripts.m37_repcount_archive import parse_official_csv, MAX_SINGLE_VIDEO_BYTES
from scripts.m37_repcount_tar import _embedded_csv
from scripts.m45_train_relative_motion_benchmark import EXPECTED_RULE
from scripts.m49_pretest_git_provenance import run as verify_m49

PROTOCOL = "m50-squat-train-annotation-video-timing-audit-v1"
COHORT_N = 20
SEED = 2026
ROOT = Path("data/processed/m50_train_label_video_integrity.json")
DEV_ROOT = Path("data/external/m45_train_relative_motion")
REPL_ROOT = Path("data/external/m46_train_rank21_40")
ARCHIVE_NAME = "RepCountA.tar.gz"
SUBJECT_FIELD_CANDIDATES = (
    "subject_id", "participant_id", "person_id",
    "athlete_id", "individual_id", "user_id"
)
SESSION_FIELD_CANDIDATES = (
    "session_id", "recording_session_id", "trial_session_id"
)
VIEW_FIELD_CANDIDATES = (
    "camera_id", "view_id", "camera_view_id"
)


def _publisher_cohorts(raw_csv: bytes):
    records = parse_official_csv(raw_csv)  # strict original annotation parser
    by_name = {row["video"].lower(): row for row in records}
    if len(by_name) != len(records):
        raise ValueError("M50 ambiguous casefolded publisher TRAIN filenames")
    salt = f"repcount-m40-train-v1:{SEED}:".encode("ascii")
    ranked = sorted(by_name, key=lambda name: (
        hashlib.sha256(salt + name.encode("ascii")).digest(), name
    ))
    if len(ranked) < 2 * COHORT_N:
        raise ValueError("M50 requires original 40 publisher Squat TRAIN sources")
    first, second = ranked[:COHORT_N], ranked[COHORT_N:2 * COHORT_N]
    if len(set(first) & set(second)):
        raise ValueError("M50 TRAIN filename cohorts unexpectedly overlap")
    return [[by_name[name] for name in first],
            [by_name[name] for name in second]]


def _header_metadata(raw_csv: bytes, cohort_names: list[list[str]]) -> dict:
    """Inspect only explicit raw TRAIN CSV columns; never infer IDs from filenames."""
    reader = csv.DictReader(io.StringIO(raw_csv.decode("utf-8-sig")))
    headers = reader.fieldnames or []
    by_key = {key.lower().strip(): key for key in headers}
    def group(candidates):
        found = [by_key[key] for key in candidates if key in by_key]
        return found[0] if len(found) == 1 else None

    selected = {
        "subject": group(SUBJECT_FIELD_CANDIDATES),
        "session": group(SESSION_FIELD_CANDIDATES),
        "camera": group(VIEW_FIELD_CANDIDATES)
    }
    groups = [set(group) for group in cohort_names]
    rows = {}
    for record in reader:
        name = str(record.get("name", "")).strip().lower()
        if name in groups[0] or name in groups[1]:
            if name in rows:
                raise ValueError("M50 duplicate selected publisher TRAIN metadata row")
            rows[name] = record
    if set(rows) != groups[0] | groups[1]:
        raise ValueError("M50 selected TRAIN metadata rows missing")

    result = {}
    for dimension, column in selected.items():
        if column is None:
            result[dimension] = {
                "recognized_explicit_column": False,
                "all_40_source_rows_populated": False,
                "across_cohort_id_overlap_verified": False,
                "observed_distinct_ids": None,
                "observed_cross_cohort_shared_ids": None,
                "interpretation": (
                    "No unique recognized explicit field. Filename patterns "
                    "are not reliable participant/session/view identifiers."
                )
            }
            continue
        samples = [
            [str(rows[name].get(column) or "").strip() for name in cohort]
            for cohort in cohort_names
        ]
        nonblank = all(value for pair in samples for value in pair)
        left, right = (set(group) - {""} for group in samples)
        result[dimension] = {
            "recognized_explicit_column": True,
            "field_name": column,  # schema only, never raw participant IDs
            "all_40_source_rows_populated": nonblank,
            "across_cohort_id_overlap_verified": False,
            "observed_distinct_ids": len(left | right) if nonblank else None,
            "observed_cross_cohort_shared_ids": (
                len(left & right) if nonblank else None
            ),
            "interpretation": (
                "Raw publisher TRAIN column values summarized, "
                "but identity semantics and independence are NOT verified"
            )
        }
    return result


def probe_video(video: Path, *, cv2_module=None) -> dict:
    """Decode frames to compare OpenCV reported FPS/count with actual frames.

    Real-video frame bytes never leave the local decoder. Injection allows
    synthetic no-video CI tests. No MediaPipe pose processing.
    """
    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError as error:
            raise RuntimeError("M50 requires OpenCV installed in local venv") from error
    if not video.is_file() or video.is_symlink():
        raise ValueError("M50 expected existing regular source TRAIN MP4")
    size = video.stat().st_size
    if not (10000 <= size <= MAX_SINGLE_VIDEO_BYTES):
        raise ValueError("M50 original selected TRAIN video bytes out of bounds")
    cap = cv2_module.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError("M50 TRAIN source cannot be opened with OpenCV")
        fps = float(cap.get(cv2_module.CAP_PROP_FPS))
        frame_claim = cap.get(cv2_module.CAP_PROP_FRAME_COUNT)
        if (not math.isfinite(fps) or not 5 <= fps <= 120 or
            not math.isfinite(frame_claim) or not 1 <= frame_claim <= fps * 121):
            raise ValueError("M50 source FPS/frame metadata invalid")
        claimed = int(frame_claim)
        decoded = 0
        # Independently count *all* decodable frames rather than stopping
        # at metadata count. A metadata discrepancy is an audit result,
        # not a reason to truncate the source video during inspection.
        hard_cap = math.ceil(fps * 121) + 3
        while decoded <= hard_cap:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
        if decoded > hard_cap:
            raise ValueError("M50 source video decode exceeds frozen 121s cap")
        if decoded <= 0:
            raise ValueError("M50 video has no decodable source frames")
        return {"fps": fps, "claimed_frames": claimed, "decoded_frames": decoded}
    finally:
        cap.release()


def _video_audit(record: dict, timing: dict) -> dict:
    frames = timing["decoded_frames"]
    claimed = timing["claimed_frames"]
    fps = timing["fps"]
    if not all(type(v) is int and v > 0 for v in (frames, claimed)):
        raise ValueError("M50 invalid video frame totals")
    if type(fps) not in (int, float) or not 5 <= fps <= 120:
        raise ValueError("M50 invalid source video FPS")
    bounds = record["cycle_bounds_frames"]
    n = record["annotated_count"]
    if n < 1 or len(bounds) != 2 * n:
        raise ValueError("M50 requires original publisher positive Squat GT")
    durations = []
    min_gap = None
    previous_end = None
    for i in range(0, len(bounds), 2):
        start, end = bounds[i:i + 2]
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError("M50 malformed publisher frame endpoint")
        if previous_end is not None:
            if start < previous_end:
                raise ValueError("M50 overlapping source publisher windows")
            gap = (start - previous_end) / fps
            min_gap = gap if min_gap is None else min(min_gap, gap)
        previous_end = end
        durations.append((end - start) / fps)
    return {
        "source_gt_reps": n,
        "claimed_frames": claimed,
        "decoded_frames": frames,
        "frame_count_difference": decoded_difference(frames, claimed),
        "label_bounds_within_decoded_video": bounds[-1] <= frames,
        "label_bounds_within_claimed_video": bounds[-1] <= claimed,
        "all_publisher_windows_shorter_than_video": all(
            end <= frames for end in bounds[1::2]
        ),
        "short_gt_intervals_lt_0_4_sec": sum(x < 0.4 for x in durations),
        "long_gt_intervals_gt_12_sec": sum(x > 12 for x in durations),
        "minimum_gt_interval_seconds": min(durations),
        "maximum_gt_interval_seconds": max(durations),
        "nonzero_temporal_overlap": False,  # already checked above
        "fps": fps,
        "annotation_video_total_seconds": frames / fps,
        "source_gt_window_total_seconds": sum(durations),
        "minimum_between_rep_gap_seconds": min_gap
    }


def decoded_difference(actual: int, metadata: int) -> int:
    return actual - metadata


def _aggregate(records: list[dict], *, split_name: str) -> dict:
    if len(records) != COHORT_N:
        raise ValueError("M50 cannot aggregate an incomplete 20-video TRAIN cohort")
    reps = sum(x["source_gt_reps"] for x in records)
    total_windows = reps
    return {
        "cohort": split_name,
        "videos": len(records),
        "annotated_reps": reps,
        "GT_interval_count": total_windows,
        "fps_min": min(x["fps"] for x in records),
        "fps_max": max(x["fps"] for x in records),
        "fps_mean": sum(x["fps"] for x in records) / len(records),
        "decoded_video_duration_min_seconds": min(
            x["annotation_video_total_seconds"] for x in records
        ),
        "decoded_video_duration_max_seconds": max(
            x["annotation_video_total_seconds"] for x in records
        ),
        "decoded_minus_claimed_frame_count_sum": sum(
            x["frame_count_difference"] for x in records
        ),
        "video_count_decode_metadata_difference": sum(
            x["frame_count_difference"] != 0 for x in records
        ),
        "videos_decoding_below_90pct_metadata": sum(
            x["decoded_frames"] < 0.9 * x["claimed_frames"] for x in records
        ),
        "videos_with_gt_bounds_exceeding_decoded_frames": sum(
            not x["label_bounds_within_decoded_video"] for x in records
        ),
        "videos_with_gt_bounds_exceeding_claimed_frames": sum(
            not x["label_bounds_within_claimed_video"] for x in records
        ),
        "short_gt_intervals_lt_0_4_sec": sum(
            x["short_gt_intervals_lt_0_4_sec"] for x in records
        ),
        "long_gt_intervals_gt_12_sec": sum(
            x["long_gt_intervals_gt_12_sec"] for x in records
        ),
        "GT_window_duration_min_seconds": min(
            x["minimum_gt_interval_seconds"] for x in records
        ),
        "GT_window_duration_max_seconds": max(
            x["maximum_gt_interval_seconds"] for x in records
        ),
        "label_window_time_coverage_seconds_total": sum(
            x["source_gt_window_total_seconds"] for x in records
        ),
        "estimated_video_duration_seconds_total": sum(
            x["annotation_video_total_seconds"] for x in records
        )
    }


def _verify_source_files(records: list[dict], root: Path) -> list[tuple[dict, Path]]:
    source = root / "official_tar_selected" / "train"
    if not source.is_dir():
        raise FileNotFoundError(
            "M50 requires previously extracted publisher TRAIN video directory"
        )
    names = {row["video"].lower() for row in records}
    available = {p.name.lower() for p in source.iterdir() if p.is_file()}
    if names != available:
        raise ValueError(
            "M50 previously extracted TRAIN filenames differ from frozen cohort"
        )
    return [(entry, source / entry["video"]) for entry in records]


def run(*, archive: Path, development_root: Path, replication_root: Path,
        m46_path: Path, m47_path: Path, m48_lock: Path, report: Path,
        probe_fn=None, source_fn=None, freeze_gate=None) -> dict:
    if report.exists():
        raise FileExistsError("M50 report exists, refusing overwrite")
    if not archive.is_file() or not archive.name.endswith((".tar.gz", ".tgz")):
        raise ValueError("M50 requires original owner RepCountA.tar.gz")
    gate = freeze_gate or verify_m49
    result = gate(
        m46_path=m46_path, m47_path=m47_path, lock_path=m48_lock,
        report_path=report, check_only=True
    )
    if result["research_source_provenance_check_passed"] is not True:
        raise ValueError("M50 requires valid original M49 source provenance gate")
    reader = source_fn or _embedded_csv
    raw = reader(archive, "train")
    sha = hashlib.sha256(raw).hexdigest()
    lock = json.loads(m48_lock.read_bytes())
    if sha != lock.get("publisher_train_annotation_sha256"):
        raise ValueError("M50 publisher TRAIN CSV SHA256 differs from M48 lock")
    cohorts = _publisher_cohorts(raw)
    if sum(x["annotated_count"] for x in cohorts[0]) != 312 or sum(
        x["annotated_count"] for x in cohorts[1]
    ) != 209:
        raise ValueError("M50 frozen first/second TRAIN GT totals disagree")
    metadata = _header_metadata(
        raw, [[r["video"].lower() for r in cohort] for cohort in cohorts]
    )
    probe = probe_fn or probe_video
    outputs = []
    for name, rows, root in (
        ("original_development_20", cohorts[0], development_root),
        ("frozen_replication_20", cohorts[1], replication_root)
    ):
        videos = _verify_source_files(rows, root)
        checked = [
            _video_audit(entry, probe(video))
            for entry, video in videos
        ]
        outputs.append(_aggregate(checked, split_name=name))
    if (outputs[0]["annotated_reps"] != 312 or
        outputs[1]["annotated_reps"] != 209):
        raise ValueError("M50 cannot reconcile original publisher TRAIN GT totals")
    problems = sum(
        d["videos_with_gt_bounds_exceeding_decoded_frames"] +
        d["videos_decoding_below_90pct_metadata"] for d in outputs
    )
    report_data = {
        "protocol": PROTOCOL,
        "study_type": "descriptive_train_only_gt_video_timing_integrity_audit",
        "source_split": "train",
        "original_m48_lock_sha256": hashlib.sha256(
            m48_lock.read_bytes()
        ).hexdigest(),
        "publisher_train_annotation_sha256": sha,
        "selection": {
            "method": "sha256_filename_hash_original_m40_seed2026",
            "first_cohort_ranks_1_to_20": True,
            "second_cohort_ranks_21_to_40": True,
            "source_filenames_disjoint": True,
            "participant_session_disjointness_unverified": True
        },
        "cohort_video_timing": outputs,
        "explicit_csv_identifier_availability": metadata,
        "videos_total": 40,
        "annotated_gt_reps_total": 521,
        "critical_timing_or_decode_integrity_flags": problems,
        "pretest_scientific_readiness": False,
        "limitations": [
            "Only TRAIN source CSV and previously extracted TRAIN clips are read.",
            "No publisher TEST/validation annotations or video contents are accessed.",
            "Video copies are locally previously extracted; no source TAR video byte parity is checked.",
            "OpenCV header FPS and constant-FPS frame-index conversion do not verify true presentation timestamps or manual GT synchronization.",
            "No participant identity is inferred from filenames. Explicit CSV ID fields, if present, have unverified semantics.",
            "A clean frame-bound check cannot establish subject-disjoint evaluation or publisher provenance.",
            "Original frozen M34/M39/M45 source code and TEST embargo are unchanged."
        ],
        "validation_or_test_scanned": False,
        "media_frames_persisted": False,
        "per_video_ids_or_metrics_persisted": False,
        "detector_or_matcher_modified": False
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf8") as handle:
        json.dump(report_data, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, default=DEV_ROOT)
    parser.add_argument("--replication-root", type=Path, default=REPL_ROOT)
    parser.add_argument("--m46-report", type=Path, default=Path(
        "data/processed/m46_train_frozen_replication.json"))
    parser.add_argument("--m47-report", type=Path, default=Path(
        "data/processed/m47_train_replication_bootstrap.json"))
    parser.add_argument("--m48-lock", type=Path, default=Path(
        "data/processed/m48_train_evidence_freeze.json"))
    parser.add_argument("--report", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        output = run(
            archive=args.source_archive,
            development_root=args.development_root,
            replication_root=args.replication_root,
            m46_path=args.m46_report,
            m47_path=args.m47_report, m48_lock=args.m48_lock,
            report=args.report
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError,
            UnicodeDecodeError, json.JSONDecodeError) as exc:
        print("M50 stopped: " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps({
        "protocol": output["protocol"],
        "videos_total": output["videos_total"],
        "annotated_gt_reps_total": output["annotated_gt_reps_total"],
        "critical_timing_or_decode_integrity_flags":
            output["critical_timing_or_decode_integrity_flags"],
        "explicit_csv_identifier_availability":
            output["explicit_csv_identifier_availability"],
        "pretest_scientific_readiness": output["pretest_scientific_readiness"],
        "validation_or_test_scanned": output["validation_or_test_scanned"],
        "report_path": str(args.report)
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
