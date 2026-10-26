#!/usr/bin/env python3
"""Sample timestamped pose features from Fitness-AQA squat videos inside a ZIP.

Only one compressed video is copied to a disposable local file at a time.
No source videos or sampled frames are written into the repository.
The publisher's clip-level train/val/test splits are preserved verbatim.

The two binary targets mean "this named error is annotated at this second",
NOT "squat is correct/incorrect", and subject identities are unknown.
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

from gym_pose.features import FEATURE_NAMES, SCHEMA_VERSION, pose_to_features
from scripts.inspect_fitness_aqa import VIDEO_EXTENSIONS, inspect, _read_json

META_COLUMNS = (
    "sample_id", "video_id", "split", "timestamp_s", "frame_index",
    "source_fps", "error_knees_forward", "error_knees_inward",
    "any_of_two_annotated_errors",
)
CSV_COLUMNS = (*META_COLUMNS, *FEATURE_NAMES)


def labels_at_time(timestamp: float, forward: list, inward: list,
                   *, margin: float = 0.1) -> tuple[int, int] | None:
    """Return two error indicators; drop samples near an annotation boundary.

    A zero denotes *absence of this named annotation*, never globally correct form.
    """
    if not math.isfinite(timestamp) or timestamp < 0 or margin < 0 or not math.isfinite(margin):
        raise ValueError("Invalid timestamp or margin")
    for intervals in (forward, inward):
        for start, end in intervals:
            if abs(timestamp - start) <= margin or abs(timestamp - end) <= margin:
                return None
    return (
        int(any(start <= timestamp <= end for start, end in forward)),
        int(any(start <= timestamp <= end for start, end in inward)),
    )


def video_members(archive: zipfile.ZipFile, allowed_keys: set[str]) -> dict[str, zipfile.ZipInfo]:
    """Find exact video stems (extra archive clips are deliberately excluded)."""
    out: dict[str, zipfile.ZipInfo] = {}
    for entry in archive.infolist():
        if entry.is_dir() or Path(entry.filename).suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        key = Path(entry.filename).stem
        if key not in allowed_keys:
            continue
        if key in out:
            raise ValueError(f"Duplicate archived video key: {key}")
        out[key] = entry
    missing = allowed_keys - set(out)
    if missing:
        raise ValueError(f"Missing source videos: {sorted(missing)[:10]}")
    return out


def _run_clip(entry: zipfile.ZipInfo, archive: zipfile.ZipFile, pose, cv2,
              writer: csv.DictWriter, *, key: str, split: str,
              forward: list, inward: list, sample_fps: float,
              margin: float, visibility: float, max_video_bytes: int) -> dict:
    counts = {
        "frames_sampled": 0,
        "pose_accepted": 0,
        "pose_rejected": 0,
        "boundary_rejected": 0,
        "annotated_knees_forward_frames": 0,
        "annotated_knees_inward_frames": 0,
    }
    # Bound temporary disk use. ZIP members are publisher-provided but still untrusted input.
    if entry.file_size > max_video_bytes:
        raise ValueError(f"{key}: uncompressed video exceeds max-video-mb allowance")
    with tempfile.TemporaryDirectory(prefix="fitness_aqa_video_") as folder:
        tmp_file = Path(folder) / ("clip" + Path(entry.filename).suffix.lower())
        with archive.open(entry) as source, tmp_file.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        reader = cv2.VideoCapture(str(tmp_file))
        try:
            if not reader.isOpened():
                raise ValueError(f"OpenCV could not decode {key}")
            fps = float(reader.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"Invalid FPS metadata for {key}: {fps}")
            frame_index = 0
            next_time = 0.0
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                now = frame_index / fps
                current_index = frame_index
                frame_index += 1
                if now + 1e-8 < next_time:
                    continue
                next_time += 1.0 / sample_fps
                counts["frames_sampled"] += 1
                labels = labels_at_time(now, forward, inward, margin=margin)
                if labels is None:
                    counts["boundary_rejected"] += 1
                    continue
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                features = (pose_to_features(result.pose_landmarks.landmark,
                                             min_visibility=visibility)
                            if result.pose_landmarks else None)
                if features is None:
                    counts["pose_rejected"] += 1
                    continue
                fwd, inw = labels
                writer.writerow({
                    "sample_id": f"{key}_f{current_index:07d}",
                    "video_id": key, "split": split,
                    "timestamp_s": round(now, 6), "frame_index": current_index,
                    "source_fps": round(fps, 5), "error_knees_forward": fwd,
                    "error_knees_inward": inw, "any_of_two_annotated_errors": int(bool(fwd or inw)),
                    **features,
                })
                counts["pose_accepted"] += 1
                counts["annotated_knees_forward_frames"] += fwd
                counts["annotated_knees_inward_frames"] += inw
        finally:
            reader.release()
    return counts


def extract(archive_path: Path, annotations: Path, output_dir: Path, *,
            selected_splits: tuple[str, ...] = ("train",), max_videos_per_split: int = 8,
            sample_fps: float = 2.0, boundary_margin: float = 0.1,
            min_visibility: float = 0.5, max_video_mb: int = 250) -> dict:
    if sample_fps <= 0 or not math.isfinite(sample_fps) or not 0 <= boundary_margin < 5:
        raise ValueError("Invalid sampling FPS or annotation boundary margin")
    if not 0 <= min_visibility <= 1 or max_videos_per_split < 0 or max_video_mb < 1:
        raise ValueError("Invalid visibility, video count or video size limit")
    if not selected_splits or any(s not in ("train", "val", "test") for s in selected_splits):
        raise ValueError("Invalid split choice")
    # Validate the entire dataset before opening videos; excludes 116 extra archive entries.
    source_summary = inspect(archive_path, annotations)
    keys_by_split = {
        split: _read_json(annotations / f"{split}_keys.json")
        for split in selected_splits
    }
    forward = _read_json(annotations / "error_knees_forward.json")
    inward = _read_json(annotations / "error_knees_inward.json")
    # Import camera libraries only when actually extracting frames (not for unit tests).
    from gym_pose.camera import _camera_libraries
    cv2, mp = _camera_libraries()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "Fitness-AQA squat labeled", "schema_version": SCHEMA_VERSION,
        "splits": {}, "sampling_fps": sample_fps,
        "boundary_margin_seconds": boundary_margin,
        "min_pose_visibility": min_visibility,
        "limit_per_split": max_videos_per_split,
        "extra_zip_videos_excluded": source_summary["extra_video_keys_count"],
        "split_type": "publisher_video_id_split_NOT_subject_disjoint",
        "target_semantics": "two named time-interval error annotations, not global correct form",
        "warnings": [
            "Sampled consecutive frames are correlated; do not randomly split them",
            "Frames without reliable pose are dropped, affecting coverage and class balance",
            "Labels zero only mean no annotation for that particular error",
            "A small pilot is not representative of the whole dataset",
        ],
    }
    with zipfile.ZipFile(archive_path) as archive:
        allowed = set().union(*(set(v) for v in keys_by_split.values()))
        members = video_members(archive, allowed)
        with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            for split in selected_splits:
                selected = sorted(keys_by_split[split])
                if max_videos_per_split > 0:
                    selected = selected[:max_videos_per_split]
                total = {
                    "selected_videos": len(selected), "videos_processed": 0,
                    "videos_failed": 0, "failure_examples": [],
                    "frames_sampled": 0, "pose_accepted": 0, "pose_rejected": 0,
                    "boundary_rejected": 0, "annotated_knees_forward_frames": 0,
                    "annotated_knees_inward_frames": 0,
                }
                result_path = output_dir / f"fitness_aqa_{split}_pose.csv"
                with result_path.open("w", newline="", encoding="utf-8") as destination:
                    writer = csv.DictWriter(destination, fieldnames=CSV_COLUMNS)
                    writer.writeheader()
                    for index, key in enumerate(selected, 1):
                        try:
                            stats = _run_clip(
                                members[key], archive, pose, cv2, writer,
                                key=key, split=split, forward=forward[key], inward=inward[key],
                                sample_fps=sample_fps, margin=boundary_margin,
                                visibility=min_visibility, max_video_bytes=max_video_mb * 1024 * 1024
                            )
                        except (OSError, ValueError, zipfile.BadZipFile) as error:
                            total["videos_failed"] += 1
                            if len(total["failure_examples"]) < 20:
                                total["failure_examples"].append({"video_id": key, "reason": str(error)})
                        else:
                            total["videos_processed"] += 1
                            for stat, count in stats.items():
                                total[stat] += count
                        if index % 25 == 0:
                            print(f"{split}: scanned {index}/{len(selected)} clips", file=sys.stderr)
                total["output_csv"] = str(result_path)
                summary["splits"][split] = total
    report_path = output_dir / "fitness_aqa_extraction_summary.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary_file"] = str(report_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path,
                        default=Path("data/external/fitness_aqa/squat/labeled/videos.zip"))
    parser.add_argument("--annotations", type=Path,
                        default=Path("data/external/fitness_aqa/squat/labeled/annotations"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/features/fitness_aqa"))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--max-videos-per-split", type=int, default=8,
                        help="Pilot limit per split; pass 0 to process all publisher-split clips")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--boundary-margin", type=float, default=0.1)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--max-video-mb", type=int, default=250)
    options = parser.parse_args()
    splits = ("train", "val", "test") if options.split == "all" else (options.split,)
    try:
        report = extract(
            options.zip, options.annotations, options.output_dir,
            selected_splits=splits, max_videos_per_split=options.max_videos_per_split,
            sample_fps=options.sample_fps, boundary_margin=options.boundary_margin,
            min_visibility=options.min_visibility, max_video_mb=options.max_video_mb
        )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Extraction failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    if any(x["videos_failed"] for x in report["splits"].values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
