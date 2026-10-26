#!/usr/bin/env python3
"""Re-extract publisher OHP train/val with distinct 30-feature elbow/knee gates.

Never decodes TEST, alters legacy 111-feature CSVs, or trains/benchmarks models.
Runs MediaPipe ONCE per annotation-eligible source frame and independently
admits each frame into elbow and/or knee task CSVs based on visible required
joints only. Safeguards partial video failures via disposable CSV spool.
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

from gym_pose.fitness_aqa_sources import inspect_ohp, read_json
from gym_pose.ohp_task_features import (
    FEATURES_BY_TASK, SCHEMA_BY_TASK, TASKS, pose_to_task_features,
)
from scripts.extract_fitness_aqa_pose import labels_at_time, video_members

META_COLUMNS = (
    "sample_id", "video_id", "split", "timestamp_s", "frame_index",
    "source_fps", "error_elbows", "error_knees", "any_of_two_annotated_errors",
)
OUTPUT_COLUMNS = {t: (*META_COLUMNS, *FEATURES_BY_TASK[t]) for t in TASKS}
COMMON_COUNTS = ("frames_sampled", "boundary_rejected", "eligible_frames",
                 "no_pose_frames", "videos_processed", "videos_failed")
TASK_COUNTS = ("accepted_frames", "rejected_frames", "positive_frames",
               "negative_frames", "videos_with_accepted_frames",
               "videos_with_positive_frames")


def _run_clip(entry, archive, pose, cv2, writers, *, key, split, elbow_windows,
              knee_windows, sample_fps, margin, min_visibility, max_video_bytes):
    counts = {
        "frames_sampled": 0, "boundary_rejected": 0, "eligible_frames": 0,
        "no_pose_frames": 0,
        "tasks": {
            task: {"accepted_frames": 0, "rejected_frames": 0, "positive_frames": 0,
                   "negative_frames": 0} for task in TASKS
        },
    }
    if entry.file_size > max_video_bytes:
        raise ValueError(f"{key}: exceeds source-video size limit")
    with tempfile.TemporaryDirectory(prefix="fitness_aqa_ohp_task_") as folder:
        clip = Path(folder) / ("clip" + Path(entry.filename).suffix.lower())
        with archive.open(entry) as reader, clip.open("wb") as dest:
            shutil.copyfileobj(reader, dest, length=1024 * 1024)
        cap = cv2.VideoCapture(str(clip))
        try:
            if not cap.isOpened():
                raise ValueError(f"{key}: video decoder failed")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"{key}: video FPS invalid")
            frame_idx, next_sample = 0, 0.
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now, index = frame_idx / fps, frame_idx
                frame_idx += 1
                if now + 1e-8 < next_sample:
                    continue
                next_sample += 1. / sample_fps
                counts["frames_sampled"] += 1
                labels = labels_at_time(now, elbow_windows, knee_windows,
                                        margin=margin)
                if labels is None:
                    counts["boundary_rejected"] += 1
                    continue
                counts["eligible_frames"] += 1
                prediction = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                points = (prediction.pose_landmarks.landmark
                          if prediction.pose_landmarks else None)
                if points is None:
                    counts["no_pose_frames"] += 1
                metadata = {
                    "sample_id": f"{key}_f{index:07d}",
                    "video_id": key, "split": split,
                    "timestamp_s": round(now, 6), "frame_index": index,
                    "source_fps": round(fps, 5),
                    "error_elbows": labels[0], "error_knees": labels[1],
                    "any_of_two_annotated_errors": int(bool(labels[0] or labels[1])),
                }
                for target, label in zip(TASKS, labels):
                    vector = (
                        pose_to_task_features(
                            points, task=target, min_visibility=min_visibility)
                        if points is not None else None
                    )
                    if vector is None:
                        counts["tasks"][target]["rejected_frames"] += 1
                        continue
                    if set(vector) != set(FEATURES_BY_TASK[target]):
                        raise ValueError(f"{key}: {target} feature schema mismatch")
                    writers[target].writerow({**metadata, **vector})
                    counts["tasks"][target]["accepted_frames"] += 1
                    counts["tasks"][target]["positive_frames" if label else "negative_frames"] += 1
        finally:
            cap.release()
    return counts


def extract(archive_path: Path, annotations: Path, output_dir: Path, *,
            selected_splits: tuple[str, ...] = ("train", "val"),
            max_videos_per_split: int = 8, sample_fps: float = 2.,
            boundary_margin: float = .1, min_visibility: float = .5,
            max_video_mb: int = 250, cv2_api=None, mp_api=None) -> dict:
    if (not selected_splits or len(set(selected_splits)) != len(selected_splits)
        or any(s not in ("train", "val") for s in selected_splits)):
        raise ValueError("Only train and validation source videos permitted; TEST locked")
    if (not math.isfinite(sample_fps) or sample_fps <= 0
        or not math.isfinite(boundary_margin) or not 0 <= boundary_margin < 5
        or not math.isfinite(min_visibility) or not 0 <= min_visibility <= 1
        or max_videos_per_split < 0 or max_video_mb < 1):
        raise ValueError("Invalid OHP task extraction settings")
    archive_path, annotations, output_dir = map(
        Path, (archive_path, annotations, output_dir))
    source = inspect_ohp(archive_path.parent, skip_archive=False)
    if not source["ready_for_pose_extraction"]:
        raise ValueError("OHP source ZIP or metadata not ready")
    selected_ids = {
        split: read_json(annotations / f"{split}_keys.json")
        for split in selected_splits
    }
    annotations_by_task = {
        "error_elbows": read_json(annotations / "error_elbows.json"),
        "error_knees": read_json(annotations / "error_knees.json"),
    }
    if cv2_api is None or mp_api is None:
        from gym_pose.camera import _camera_libraries
        cv2_api, mp_api = _camera_libraries()
    summary = {
        "status": "OHP_task_specific_pose_extraction_NO_TEST",
        "dataset": "Fitness-AQA OHP publisher train-val",
        "source_split_type": "publisher_video_ID_disjoint_NOT_subject_disjoint",
        "schema_by_task": SCHEMA_BY_TASK, "feature_count_by_task": {
            task: len(FEATURES_BY_TASK[task]) for task in TASKS},
        "source_test_videos_processed": 0,
        "extra_source_videos_excluded": source["archive"]["extra_keys_count"],
        "sampling_fps": sample_fps, "boundary_margin_seconds": boundary_margin,
        "min_visibility_on_required_joints": min_visibility,
        "limit_per_split": max_videos_per_split,
        "splits": {},
        "warnings": [
            "Separate per-task feature schema and visibility gate; do NOT combine with the 111-feature OHP baseline without a matched-frame audit",
            "Elbow model uses shoulders, elbows, wrists and hips; knee model uses shoulders, hips, knees and ankles",
            "A missing/unreliable pose is an abstention, not a true negative",
            "Zero error annotation does not establish globally correct pressing technique",
            "Publisher clip IDs do not establish subject-independent evaluation",
            "No training, validation metrics, or test evaluation in this extraction",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        allowed = set().union(*(set(ids) for ids in selected_ids.values()))
        members = video_members(archive, allowed)
        with mp_api.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            for split in selected_splits:
                selected = sorted(selected_ids[split])
                if max_videos_per_split:
                    selected = selected[:max_videos_per_split]
                counts = {
                    "selected_videos": len(selected), "videos_processed": 0,
                    "videos_failed": 0, "failure_examples": [],
                    "frames_sampled": 0, "boundary_rejected": 0,
                    "eligible_frames": 0, "no_pose_frames": 0,
                    "tasks": {task: {stat: 0 for stat in TASK_COUNTS} for task in TASKS},
                }
                destinations = {
                    task: output_dir / f"fitness_aqa_ohp_{task}_{split}_pose.csv"
                    for task in TASKS
                }
                from contextlib import ExitStack
                with ExitStack() as stack:
                    dest = {
                        task: stack.enter_context(path.open(
                            "w", newline="", encoding="utf-8"))
                        for task, path in destinations.items()
                    }
                    for task in TASKS:
                        csv.DictWriter(dest[task], fieldnames=OUTPUT_COLUMNS[task]).writeheader()
                    for idx, key in enumerate(selected, 1):
                        with ExitStack() as temporary:
                            streams = {
                                task: temporary.enter_context(tempfile.TemporaryFile(
                                    mode="w+t", newline="", encoding="utf-8"))
                                for task in TASKS
                            }
                            writers = {
                                task: csv.DictWriter(streams[task], fieldnames=OUTPUT_COLUMNS[task])
                                for task in TASKS
                            }
                            try:
                                clip_stats = _run_clip(
                                    members[key], archive, pose, cv2_api, writers,
                                    key=key, split=split,
                                    elbow_windows=annotations_by_task["error_elbows"][key],
                                    knee_windows=annotations_by_task["error_knees"][key],
                                    sample_fps=sample_fps, margin=boundary_margin,
                                    min_visibility=min_visibility,
                                    max_video_bytes=max_video_mb * 1024 * 1024,
                                )
                            except (OSError, ValueError, TypeError, zipfile.BadZipFile) as exc:
                                counts["videos_failed"] += 1
                                if len(counts["failure_examples"]) < 15:
                                    counts["failure_examples"].append({
                                        "video_id": key, "reason": str(exc)})
                            else:
                                counts["videos_processed"] += 1
                                for field in ("frames_sampled", "boundary_rejected",
                                              "eligible_frames", "no_pose_frames"):
                                    counts[field] += clip_stats[field]
                                for task in TASKS:
                                    for metric in ("accepted_frames", "rejected_frames",
                                                   "positive_frames", "negative_frames"):
                                        counts["tasks"][task][metric] += clip_stats["tasks"][task][metric]
                                    if clip_stats["tasks"][task]["accepted_frames"]:
                                        counts["tasks"][task]["videos_with_accepted_frames"] += 1
                                    if clip_stats["tasks"][task]["positive_frames"]:
                                        counts["tasks"][task]["videos_with_positive_frames"] += 1
                                    streams[task].seek(0)
                                    shutil.copyfileobj(streams[task], dest[task])
                        if idx % 25 == 0:
                            print(f"{split}: processed {idx}/{len(selected)} clips", file=sys.stderr)
                for task in TASKS:
                    stats = counts["tasks"][task]
                    stats["pose_coverage_eligible"] = (
                        stats["accepted_frames"] / counts["eligible_frames"]
                        if counts["eligible_frames"] else None)
                    stats["csv"] = str(destinations[task])
                counts["any_frame_policy_difference"] = True
                summary["splits"][split] = counts
    summary_path = output_dir / "fitness_aqa_ohp_task_extraction_summary.json"
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
        "data/features/fitness_aqa_ohp_tasks"))
    parser.add_argument("--split", choices=("train","val","train-val"), default="train-val")
    parser.add_argument("--max-videos-per-split", type=int, default=8)
    parser.add_argument("--sample-fps", type=float, default=2.)
    parser.add_argument("--boundary-margin", type=float, default=.1)
    parser.add_argument("--min-visibility", type=float, default=.5)
    parser.add_argument("--max-video-mb", type=int, default=250)
    args = parser.parse_args()
    selected = ("train","val") if args.split == "train-val" else (args.split,)
    try:
        report = extract(args.zip, args.annotations, args.output_dir,
                         selected_splits=selected,
                         max_videos_per_split=args.max_videos_per_split,
                         sample_fps=args.sample_fps,
                         boundary_margin=args.boundary_margin,
                         min_visibility=args.min_visibility,
                         max_video_mb=args.max_video_mb)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError,
            KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(f"OHP task extraction failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    if any(s["videos_failed"] for s in report["splits"].values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
