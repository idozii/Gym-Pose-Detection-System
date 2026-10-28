"""M52: TRAIN-only video presentation-timestamp vs frame/FPS timeline audit.

Inspect 40 original author TRAIN Squat videos already extracted in M45/M46.
Compare FFprobe per-frame best-effort presentation timestamps to the
frame_index / OpenCV fps timing used by original M37-M47 GT matching.

This is a diagnostic of VIDEO CLOCK assumptions, NOT a validation of
human-labeled squat event boundaries, a new benchmark, or TEST execution.
The M48-frozen 17 source files are untouched. No MediaPipe, JS replay,
landmarks, or per-video timings are saved. All probes are local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from scripts.m37_repcount_archive import MAX_SINGLE_VIDEO_BYTES
from scripts.m37_repcount_tar import _embedded_csv
from scripts.m49_pretest_git_provenance import run as pretest_git_gate
from scripts.m50_train_label_video_integrity import (
    _publisher_cohorts, _verify_source_files,
)
from scripts.m51_verify_train_archive_byte_parity import (
    _verify_existing_m50
)

PROTOCOL = "m52-train-frame-pts-vs-fps-audit-v1"
REPO = Path(__file__).resolve().parents[1]
DEFAULT_DEV = Path("data/external/m45_train_relative_motion")
DEFAULT_REPL = Path("data/external/m46_train_rank21_40")
DEFAULT_M50 = Path("data/processed/m50_train_label_video_integrity.json")
DEFAULT_M51 = Path("data/processed/m51_train_archive_byte_parity.json")
DEFAULT_REPORT = Path("data/processed/m52_train_pts_timeline_audit.json")
FPS_DRIFT_GT_END_THRESHOLD_MS = 50.0
IRREGULAR_FRAME_STEP_TOLERANCE = 0.20
MAX_FFPROBE_FRAMES = 15000


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ensure_prior_m51(
    *, m51_path: Path, m50_path: Path, m48_bytes: bytes, csv_sha: str
) -> dict:
    if not m51_path.is_file():
        raise FileNotFoundError("M52 requires previous owner-run M51 source parity JSON")
    m51 = json.loads(m51_path.read_bytes())
    if not isinstance(m51, dict):
        raise ValueError("M52 original M51 report must be an object")
    expected = (
        ("original_development_20", 20, 312),
        ("frozen_replication_20", 20, 209)
    )
    entries = m51.get("cohorts")
    if (
        m51.get("protocol") != "m51-train-selected-archive-byte-parity-v1" or
        m51.get("publisher_train_annotation_sha256") != csv_sha or
        m51.get("original_m48_lock_sha256") != _sha(m48_bytes) or
        m51.get("original_m50_report_sha256") != _sha(m50_path.read_bytes()) or
        m51.get("source_split_video_contents_hashed") != "train_only" or
        m51.get("videos_byte_identical_total") != 40 or
        m51.get("annotated_gt_reps_total") != 521 or
        m51.get("verification_status") !=
            "all_40_train_local_mp4_bytes_equal_tar_members" or
        m51.get("test_evaluation_performed") is not False or
        m51.get("validation_or_test_video_payload_hashed_or_decoded") is not False or
        not isinstance(entries, list) or len(entries) != 2 or
        tuple(
            (x.get("cohort"), x.get("videos_byte_identical_to_source_tar"),
             x.get("annotated_gt_reps"))
            for x in entries if isinstance(x, dict)
        ) != expected
    ):
        raise ValueError("M52 previous M51 40-TRAIN original-source evidence mismatch")
    return m51


def probe_pts(video: Path, *, runner=None) -> list[float]:
    """Return presentation times in decoded/presentation order, in seconds.

    FFprobe '-show_frames' is queried for v:0 only. The command output is
    not persisted. Probe timestamps are not independently measured clock
    times; they are values derived from container/decoder metadata.
    """
    if not video.is_file() or video.is_symlink() or not (
        10000 <= video.stat().st_size <= MAX_SINGLE_VIDEO_BYTES
    ):
        raise ValueError("M52 requires pre-existing regular TRAIN MP4 file")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "json", "-i", str(video)
    ]
    invoke = runner or subprocess.run
    try:
        result = invoke(
            cmd, check=False, capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "M52 requires installed FFprobe (ffmpeg package) and readable TRAIN MP4"
        ) from exc
    if result.returncode:
        raise RuntimeError("M52 FFprobe failed for an existing selected TRAIN clip")
    if len(result.stdout) > 8_000_000:
        raise ValueError("M52 FFprobe output exceeds bounded metadata limit")
    raw = json.loads(result.stdout)
    if not isinstance(raw, dict) or not isinstance(raw.get("frames"), list):
        raise ValueError("M52 FFprobe did not return a video FRAME series")
    rows = raw["frames"]
    if not 8 <= len(rows) <= MAX_FFPROBE_FRAMES:
        raise ValueError("M52 invalid number of FFprobe frame timestamps")
    output = []
    previous = -math.inf
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("M52 malformed FFprobe frame record")
        raw_time = row.get("best_effort_timestamp_time")
        if not isinstance(raw_time, str):
            raise ValueError("M52 missing presentation timestamp")
        try:
            timestamp = float(raw_time)
        except (ValueError, OverflowError) as exc:
            raise ValueError("M52 nonnumeric video presentation timestamp") from exc
        if not math.isfinite(timestamp) or timestamp <= previous:
            raise ValueError("M52 PTS missing, nonfinite or not strictly increasing")
        output.append(timestamp)
        previous = timestamp
    return output


def audit_timeline(*, gt_bounds: list[int], fps: float,
                   declared_frames: int, pts: list[float]) -> dict:
    """Use original exclusive GT end coordinates, WITHOUT rematching events."""
    if (type(fps) not in (float, int) or not math.isfinite(fps) or
        not 5 <= fps <= 120 or type(declared_frames) is not int or
        not 8 <= declared_frames <= MAX_FFPROBE_FRAMES or
        not isinstance(gt_bounds, list) or len(gt_bounds) == 0 or
        len(gt_bounds) % 2 or
        not all(type(x) is int for x in gt_bounds)):
        raise ValueError("M52 requires frozen valid fps/GT frame-index metadata")
    if len(pts) != declared_frames:
        raise ValueError("M52 FFprobe frame count differs from M50 OpenCV decoded count")
    if any(not math.isfinite(x) for x in pts):
        raise ValueError("M52 invalid presentation timestamp")
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("M52 presentation times must be strictly increasing")
    if not all(0 <= a < b <= declared_frames for a, b in zip(
        gt_bounds[::2], gt_bounds[1::2]
    )):
        raise ValueError("M52 GT endpoint exceeds original decoded video")
    first = pts[0]
    nominal_step = 1 / fps
    increments = [b - a for a, b in zip(pts, pts[1:])]
    median_step = statistics.median(increments)
    if median_step <= 0:
        raise ValueError("M52 invalid median frame interval")
    irregular = sum(
        abs(x - median_step) > IRREGULAR_FRAME_STEP_TOLERANCE * median_step
        for x in increments
    )
    expected_relative = [(i * nominal_step) for i in range(len(pts))]
    drift = [
        ((p - first) - theoretical) * 1000
        for p, theoretical in zip(pts, expected_relative)
    ]
    # GT interval is [start, end); its end is an *exclusive* index.
    # For end==N, there is NO observed displayed frame at index N,
    # so never pretend an extrapolation is an observed PTS.
    boundary_errors = []
    unavailable_ends = 0
    for start, end in zip(gt_bounds[::2], gt_bounds[1::2]):
        boundary_errors.append(abs(drift[start]))
        if end < declared_frames:
            boundary_errors.append(abs(drift[end]))
        else:
            unavailable_ends += 1
    over = sum(x > FPS_DRIFT_GT_END_THRESHOLD_MS for x in boundary_errors)
    return {
        "pts_frames": len(pts),
        "GT_windows": len(gt_bounds) // 2,
        "first_presentation_timestamp_seconds": first,
        "median_frame_step_seconds": median_step,
        "nominal_frame_step_seconds": nominal_step,
        "PTS_last_minus_first_seconds": pts[-1] - first,
        "nominal_last_frame_offset_seconds": (len(pts)-1) / fps,
        "max_abs_timeline_drift_ms": max(abs(x) for x in drift),
        "max_abs_GT_boundary_timeline_drift_ms": max(boundary_errors),
        "GT_boundaries_with_abs_drift_gt50ms": over,
        "observable_GT_boundary_points": len(boundary_errors),
        "exclusive_GT_ends_without_corresponding_PTS_frame": unavailable_ends,
        "irregular_interframe_steps_gt20pct_from_median": irregular,
        "median_derived_fps_deviation_pct": abs(
            (1 / median_step) / fps - 1
        ) * 100
    }


def _get_original_video_fps_and_frames(path: Path, *, cv2_module=None):
    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise RuntimeError("M52 requires existing OpenCV library") from exc
    cap = cv2_module.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("M52 source video not readable by OpenCV")
        fps = float(cap.get(cv2_module.CAP_PROP_FPS))
        count = float(cap.get(cv2_module.CAP_PROP_FRAME_COUNT))
        if (not math.isfinite(fps) or not 5 <= fps <= 120 or
            not math.isfinite(count) or
            not 8 <= count <= MAX_FFPROBE_FRAMES):
            raise ValueError("M52 original decoded source FPS/count invalid")
        return fps, int(count)
    finally:
        cap.release()


def _cohort(rows: list[dict], *, name: str) -> dict:
    if len(rows) != 20:
        raise ValueError("M52 requires exactly 20 videos in each frozen TRAIN cohort")
    return {
        "cohort": name,
        "videos": len(rows),
        "GT_reps": sum(x["GT_windows"] for x in rows),
        "ffprobe_frame_count_total": sum(x["pts_frames"] for x in rows),
        "videos_with_irregular_interframe_PTS": sum(
            x["irregular_interframe_steps_gt20pct_from_median"] > 0 for x in rows
        ),
        "irregular_interframe_steps_total": sum(
            x["irregular_interframe_steps_gt20pct_from_median"] for x in rows
        ),
        "videos_with_any_gt_boundary_drift_gt50ms": sum(
            x["GT_boundaries_with_abs_drift_gt50ms"] > 0 for x in rows
        ),
        "GT_boundary_points_with_abs_drift_gt50ms": sum(
            x["GT_boundaries_with_abs_drift_gt50ms"] for x in rows
        ),
        "GT_boundary_points_observable": sum(
            x["observable_GT_boundary_points"] for x in rows
        ),
        "exclusive_gt_ends_not_directly_timestamped": sum(
            x["exclusive_GT_ends_without_corresponding_PTS_frame"] for x in rows
        ),
        "max_video_abs_timeline_drift_ms": max(
            x["max_abs_timeline_drift_ms"] for x in rows
        ),
        "max_abs_gt_boundary_timeline_drift_ms": max(
            x["max_abs_GT_boundary_timeline_drift_ms"] for x in rows
        ),
        "median_derived_fps_deviation_max_pct": max(
            x["median_derived_fps_deviation_pct"] for x in rows
        ),
        "nonzero_initial_presentation_offset_videos": sum(
            abs(x["first_presentation_timestamp_seconds"]) > 1e-8 for x in rows
        )
    }


def run(*, archive: Path, development_root: Path, replication_root: Path,
        m46_path: Path, m47_path: Path, m48_lock: Path,
        m50_path: Path, m51_path: Path, report: Path,
        csv_reader=None, pts_probe=None, fps_probe=None,
        evidence_gate=None) -> dict:
    if report.exists():
        raise FileExistsError("M52 report exists; refusing overwrite")
    gate = evidence_gate or pretest_git_gate
    result = gate(
        m46_path=m46_path, m47_path=m47_path, lock_path=m48_lock,
        report_path=report, check_only=True
    )
    if result.get("research_source_provenance_check_passed") is not True:
        raise ValueError("M52 original M48/M49 evidence source gate failed")
    lock_bytes = m48_lock.read_bytes()
    lock = json.loads(lock_bytes)
    if not archive.is_file() or not archive.name.lower().endswith((".tar.gz",".tgz")):
        raise ValueError("M52 requires original local publisher TAR.GZ file")
    read_train_csv = csv_reader or _embedded_csv
    raw = read_train_csv(archive, "train")
    train_sha = _sha(raw)
    if train_sha != lock.get("publisher_train_annotation_sha256"):
        raise ValueError("M52 original TRAIN CSV SHA256 disagrees with M48 lock")
    _verify_existing_m50(
        m50_path, publisher_sha=train_sha,
        original_lock_sha=_sha(lock_bytes)
    )
    _ensure_prior_m51(
        m51_path=m51_path, m50_path=m50_path, m48_bytes=lock_bytes,
        csv_sha=train_sha
    )
    groups = _publisher_cohorts(raw)
    if [sum(x["annotated_count"] for x in g) for g in groups] != [312,209]:
        raise ValueError("M52 original 40-TRAIN GT totals differ from locked evidence")
    probe_pts_fn = pts_probe or probe_pts
    probe_cv_fn = fps_probe or _get_original_video_fps_and_frames
    output = []
    for label, cohort, root in (
        ("original_development_20", groups[0], development_root),
        ("frozen_replication_20", groups[1], replication_root)
    ):
        videos = _verify_source_files(cohort, root)
        measures = []
        for record, path in videos:
            fps, frames = probe_cv_fn(path)
            if frames < record["cycle_bounds_frames"][-1]:
                raise ValueError("M52 GT index exceeds original frame count")
            measured_pts = probe_pts_fn(path)
            measures.append(audit_timeline(
                gt_bounds=record["cycle_bounds_frames"],
                fps=fps, declared_frames=frames, pts=measured_pts
            ))
        output.append(_cohort(measures, name=label))
    if [x["GT_reps"] for x in output] != [312,209]:
        raise ValueError("M52 GT event totals cannot be reconciled")
    audit = {
        "protocol": PROTOCOL,
        "split": "train",
        "source": "owner-provided TAR.GZ original TRAIN CSV and previously extracted source TRAIN MP4s",
        "author_source_provenance_licensed_or_verified": False,
        "publisher_train_annotation_sha256": train_sha,
        "original_m48_lock_sha256": _sha(lock_bytes),
        "original_m50_report_sha256": _sha(m50_path.read_bytes()),
        "original_m51_report_sha256": _sha(m51_path.read_bytes()),
        "cohort_video_timeline": output,
        "videos_evaluated": 40,
        "annotated_gt_events": 521,
        "timing_thresholds": {
            "gt_boundary_nominal_vs_PTS_drift_ms": FPS_DRIFT_GT_END_THRESHOLD_MS,
            "interframe_duration_relative_to_median": IRREGULAR_FRAME_STEP_TOLERANCE,
            "PTS_clock_reference": "relative_to_first_video_frame",
            "source_frame_index_time": "frame_index / OpenCV_fps",
            "GT_intervals": "original_publisher_half_open_[start,end)"
        },
        "PTS_timeline_anomalies_descriptive": sum(
            x["videos_with_irregular_interframe_PTS"] +
            x["videos_with_any_gt_boundary_drift_gt50ms"] for x in output
        ),
        "limits": [
            "FFprobe best_effort_timestamp_time is decoder/container-derived, not independent wall clock truth.",
            "Frame PTS-vs-nominal checks do not show whether manually published repetition GT windows are human-aligned.",
            "Original GT end==frame_count has no observed end-index PTS; it is recorded as unavailable instead of extrapolated.",
            "Only original source TRAIN CSV and already extracted 40 TRAIN MP4s are examined, never TEST or validation annotations/videos.",
            "No per-video names, GT intervals, actual PTS arrays or pose landmarks persist.",
            "The existing M51 result is cross-checked as a prior report, but individual local videos are not rehashed against TAR by M52.",
            "Video filename disjointness is not participant/session/camera disjointness.",
            "Original M37/M39/M45 counter and 17 M48-frozen source files unchanged; no TEST metric computed."
        ],
        "test_or_validation_media_scanned": False,
        "test_benchmark_executed": False,
        "deployed_counter_modified": False,
        "per_frame_data_persisted": False
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf8") as dest:
        json.dump(audit, dest, indent=2, allow_nan=False)
        dest.write("\n")
    return audit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--development-root", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--replication-root", type=Path, default=DEFAULT_REPL)
    parser.add_argument("--m46-report", type=Path, default=Path(
        "data/processed/m46_train_frozen_replication.json"))
    parser.add_argument("--m47-report", type=Path, default=Path(
        "data/processed/m47_train_replication_bootstrap.json"))
    parser.add_argument("--m48-lock", type=Path, default=Path(
        "data/processed/m48_train_evidence_freeze.json"))
    parser.add_argument("--m50-report", type=Path, default=DEFAULT_M50)
    parser.add_argument("--m51-report", type=Path, default=DEFAULT_M51)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    a = parser.parse_args(argv)
    try:
        output = run(
            archive=a.source_archive,
            development_root=a.development_root,
            replication_root=a.replication_root,
            m46_path=a.m46_report, m47_path=a.m47_report,
            m48_lock=a.m48_lock, m50_path=a.m50_report,
            m51_path=a.m51_report, report=a.report
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError,
            UnicodeDecodeError, json.JSONDecodeError) as exc:
        print("M52 stopped: "+str(exc), file=sys.stderr)
        return 1
    print(json.dumps({
        "protocol":output["protocol"],
        "videos_evaluated":output["videos_evaluated"],
        "annotated_gt_events":output["annotated_gt_events"],
        "PTS_timeline_anomalies_descriptive":output["PTS_timeline_anomalies_descriptive"],
        "test_or_validation_media_scanned":False,
        "report":str(a.report)
    }, indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
