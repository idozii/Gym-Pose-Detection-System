#!/usr/bin/env python3
"""Pilot / full source TRAIN+VAL Fitness-AQA overhead press pose extraction.

Use only original publisher clip IDs and named elbow/knee annotation intervals.
The original source TEST clips are never decoded or sampled by this script.
Does not retrain models, select thresholds, or claim human-independent evaluation.
ZIP video members are streamed one at a time to bounded temporary storage.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from gym_pose.ohp_features import (
    OHP_FEATURE_NAMES, OHP_SCHEMA_VERSION, pose_to_ohp_features,
)
from gym_pose.fitness_aqa_sources import inspect_ohp, read_json
from scripts.extract_fitness_aqa_pose import labels_at_time, video_members

OHP_TARGETS = ("error_elbows", "error_knees")
OHP_META_COLUMNS = (
    "sample_id", "video_id", "split", "timestamp_s", "frame_index", "source_fps",
    "error_elbows", "error_knees", "any_of_two_annotated_errors",
)
OHP_CSV_COLUMNS = (*OHP_META_COLUMNS, *OHP_FEATURE_NAMES)
PER_CLIP_COUNTERS = (
    "frames_sampled", "pose_accepted", "pose_rejected", "boundary_rejected",
    "annotated_error_elbows_frames", "annotated_error_knees_frames",
)


def run_clip(entry: zipfile.ZipInfo, archive: zipfile.ZipFile, pose, cv2,
             writer: csv.DictWriter, *, video_id: str, split: str,
             elbow_windows: list, knee_windows: list,
             sample_fps: float, boundary_margin: float,
             min_visibility: float, max_video_bytes: int) -> dict:
    """Process one source clip. Caller must discard writer on failed clips."""
    counts = {key: 0 for key in PER_CLIP_COUNTERS}
    if entry.file_size > max_video_bytes:
        raise ValueError(f"{video_id}: video exceeds max_video_bytes limit")
    with tempfile.TemporaryDirectory(prefix="fitness_aqa_ohp_") as directory:
        video_path = Path(directory) / ("clip" + Path(entry.filename).suffix.lower())
        with archive.open(entry) as source, video_path.open("wb") as dest:
            shutil.copyfileobj(source, dest, length=1024 * 1024)
        reader = cv2.VideoCapture(str(video_path))
        try:
            if not reader.isOpened():
                raise ValueError(f"{video_id}: cannot decode video")
            fps = float(reader.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"{video_id}: invalid source FPS {fps}")
            frame_index, next_time = 0, 0.0
            while True:
                read_ok, frame = reader.read()
                if not read_ok:
                    break
                now = frame_index / fps
                current_index = frame_index
                frame_index += 1
                if now + 1e-8 < next_time:
                    continue
                next_time += 1.0 / sample_fps
                counts["frames_sampled"] += 1
                # Existing function uses interval membership ('any'), NOT
                # timestamp ordering. Source OHP intervals can be unsorted.
                labels = labels_at_time(
                    now, elbow_windows, knee_windows, margin=boundary_margin)
                if labels is None:
                    counts["boundary_rejected"] += 1
                    continue
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmark_set = (
                    result.pose_landmarks.landmark if result.pose_landmarks else None
                )
                feature_map = (
                    pose_to_ohp_features(landmark_set, min_visibility=min_visibility)
                    if landmark_set is not None else None
                )
                if feature_map is None:
                    counts["pose_rejected"] += 1
                    continue
                if set(feature_map) != set(OHP_FEATURE_NAMES):
                    raise ValueError(f"{video_id}: OHP feature schema mismatch")
                elbows, knees = labels
                writer.writerow({
                    "sample_id": f"{video_id}_f{current_index:07d}",
                    "video_id": video_id, "split": split,
                    "timestamp_s": round(now, 6),
                    "frame_index": current_index, "source_fps": round(fps, 5),
                    "error_elbows": elbows, "error_knees": knees,
                    "any_of_two_annotated_errors": int(bool(elbows or knees)),
                    **feature_map,
                })
                counts["pose_accepted"] += 1
                counts["annotated_error_elbows_frames"] += elbows
                counts["annotated_error_knees_frames"] += knees
        finally:
            reader.release()
    return counts


def extract(archive_path: Path, annotations: Path, output_dir: Path, *,
            selected_splits: tuple[str, ...] = ("train", "val"),
            max_videos_per_split: int = 8, sample_fps: float = 2.0,
            boundary_margin: float = .1, min_visibility: float = .5,
            max_video_mb: int = 250, cv2_api=None, mp_api=None) -> dict:
    if (not selected_splits or len(set(selected_splits)) != len(selected_splits)
        or any(split not in ("train", "val") for split in selected_splits)):
        raise ValueError("Only original TRAIN and VAL may be extracted; test is locked")
    if (not math.isfinite(sample_fps) or sample_fps <= 0
        or not math.isfinite(boundary_margin) or not 0 <= boundary_margin < 5
        or not 0 <= min_visibility <= 1
        or max_videos_per_split < 0 or max_video_mb < 1):
        raise ValueError("Invalid extraction sampling/visibility/video size parameters")
    archive_path, annotations, output_dir = (
        Path(archive_path), Path(annotations), Path(output_dir)
    )
    dataset = inspect_ohp(archive_path.parent, skip_archive=False)
    if not dataset["ready_for_pose_extraction"]:
        raise ValueError("OHP ZIP metadata not complete: missing or duplicated source clips")
    keys = {
        split: read_json(annotations / f"{split}_keys.json")
        for split in selected_splits
    }
    elbows = read_json(annotations / "error_elbows.json")
    knees = read_json(annotations / "error_knees.json")
    if cv2_api is None or mp_api is None:
        from gym_pose.camera import _camera_libraries
        cv2_api, mp_api = _camera_libraries()
    summary = {
        "dataset": "Fitness-AQA overhead press labeled",
        "exercise": "ohp", "feature_schema_version": OHP_SCHEMA_VERSION,
        "n_features": len(OHP_FEATURE_NAMES), "targets": list(OHP_TARGETS),
        "sampling_fps": sample_fps, "boundary_margin_seconds": boundary_margin,
        "min_pose_visibility": min_visibility, "limit_per_split": max_videos_per_split,
        "source_split_type": "publisher_video_ID_disjoint_NOT_subject_disjoint",
        "extracted_splits": list(selected_splits),
        "source_test_clips_processed": 0,
        "source_extra_archive_video_ids_excluded": dataset["archive"]["extra_keys_count"],
        "splits": {},
        "warnings": [
            "Positive labels mean a named error is annotated at this timestamp",
            "Negative labels do not mean the entire OHP technique is correct",
            "OHP feature geometry is camera-relative, not medically validated",
            "Source subject identities are unknown; only publisher clip IDs are disjoint",
            "Frames without reliable OHP joints are excluded; quantify pose coverage",
            "Original source test video was not decoded, sampled, or evaluated",
            "Pilot first, then full original TRAIN+VAL without updating model thresholds from validation",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        allowed = set().union(*(set(v) for v in keys.values()))
        members = video_members(archive, allowed)
        with mp_api.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            for split in selected_splits:
                selected = sorted(keys[split])
                if max_videos_per_split:
                    selected = selected[:max_videos_per_split]
                total = {
                    "selected_videos": len(selected), "videos_processed": 0,
                    "videos_failed": 0, "failure_examples": [],
                    **{name: 0 for name in PER_CLIP_COUNTERS},
                }
                output_csv = output_dir / f"fitness_aqa_ohp_{split}_pose.csv"
                with output_csv.open("w", newline="", encoding="utf-8") as dest:
                    writer = csv.DictWriter(dest, fieldnames=OHP_CSV_COLUMNS)
                    writer.writeheader()
                    for n, video_id in enumerate(selected, start=1):
                        # Spool each clip's CSV: broken clips never leave partial
                        # source rows in the output or inflate class counts.
                        with tempfile.TemporaryFile(mode="w+t", newline="", encoding="utf-8") as spool:
                            clip_writer = csv.DictWriter(
                                spool, fieldnames=OHP_CSV_COLUMNS)
                            try:
                                counts = run_clip(
                                    members[video_id], archive, pose, cv2_api,
                                    clip_writer, video_id=video_id, split=split,
                                    elbow_windows=elbows[video_id],
                                    knee_windows=knees[video_id],
                                    sample_fps=sample_fps,
                                    boundary_margin=boundary_margin,
                                    min_visibility=min_visibility,
                                    max_video_bytes=max_video_mb * 1024 * 1024,
                                )
                            except (OSError, ValueError, TypeError, zipfile.BadZipFile) as exc:
                                total["videos_failed"] += 1
                                if len(total["failure_examples"]) < 15:
                                    total["failure_examples"].append({
                                        "video_id": video_id, "reason": str(exc)
                                    })
                            else:
                                spool.seek(0)
                                shutil.copyfileobj(spool, dest)
                                total["videos_processed"] += 1
                                for metric in PER_CLIP_COUNTERS:
                                    total[metric] += counts[metric]
                        if n % 25 == 0:
                            print(f"{split}: processed {n}/{len(selected)} clips", file=sys.stderr)
                eligible = total["frames_sampled"] - total["boundary_rejected"]
                total["eligible_sampled_frames"] = eligible
                total["pose_coverage_eligible"] = (
                    total["pose_accepted"] / eligible if eligible else None
                )
                total["output_csv"] = str(output_csv)
                summary["splits"][split] = total
    summary_path = output_dir / "fitness_aqa_ohp_extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary_file"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path(
        "data/external/fitness_aqa/ohp/labeled/videos.zip"))
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/external/fitness_aqa/ohp/labeled/annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/features/fitness_aqa_ohp"))
    parser.add_argument("--split", choices=("train", "val", "train-val"), default="train-val")
    parser.add_argument("--max-videos-per-split", type=int, default=8,
                        help="8 videos each by default; pass 0 to process all original source clips")
    parser.add_argument("--sample-fps", type=float, default=2.)
    parser.add_argument("--boundary-margin", type=float, default=.1)
    parser.add_argument("--min-visibility", type=float, default=.5)
    parser.add_argument("--max-video-mb", type=int, default=250)
    args = parser.parse_args()
    splits = ("train", "val") if args.split == "train-val" else (args.split,)
    try:
        summary = extract(
            args.zip, args.annotations, args.output_dir,
            selected_splits=splits, max_videos_per_split=args.max_videos_per_split,
            sample_fps=args.sample_fps, boundary_margin=args.boundary_margin,
            min_visibility=args.min_visibility, max_video_mb=args.max_video_mb,
        )
    except (OSError, ValueError, RuntimeError, KeyError, ImportError,
            zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(f"OHP extraction failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    if any(stat["videos_failed"] for stat in summary["splits"].values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
