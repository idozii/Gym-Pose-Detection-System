#!/usr/bin/env python3
"""Diagnose why Fitness-AQA squat pose frames fail the production quality gate.

A reproducible, annotation-stratified sample of SOURCE train/val video clips is
read one clip at a time from the existing ZIP. The script does NOT read test
clips, train models, modify previously extracted CSVs, or save source images.
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
from collections import Counter
from pathlib import Path

import numpy as np

from gym_pose.quality import JOINT_NAMES, REASONS, assess_pose_quality
from scripts.extract_fitness_aqa_pose import labels_at_time, video_members
from scripts.inspect_fitness_aqa import _read_json

VISIBILITY_THRESHOLDS = (0.25, 0.35, 0.50, 0.65)
CSV_FIELDS = (
    "split", "video_id", "video_annotation_stratum", "frame_index",
    "timestamp_s", "error_knees_forward", "error_knees_inward",
    "rejection_reason", "low_visibility_joints",
    "minimum_critical_joint_visibility", "shoulder_distance",
)


def annotation_stratum(forward_intervals: list, inward_intervals: list) -> str:
    if inward_intervals:
        return "inward_annotated_video"
    if forward_intervals:
        return "forward_only_annotated_video"
    return "neither_of_two_annotated_errors"


def stratified_video_sample(keys: list[str], forward: dict, inward: dict, *,
                            count: int, seed: int) -> list[tuple[str, str]]:
    """Balanced across available video annotation strata for diagnostic coverage.

    Not proportional to dataset prevalence; summary must never be extrapolated
    into an estimate of full-dataset acceptance without sampling weights.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if len(keys) != len(set(keys)) or any(k not in forward or k not in inward for k in keys):
        raise ValueError("Duplicate/missing annotation video IDs")
    labels = ("inward_annotated_video", "forward_only_annotated_video",
              "neither_of_two_annotated_errors")
    groups = {s: [] for s in labels}
    for key in sorted(keys):
        groups[annotation_stratum(forward[key], inward[key])].append(key)
    rng = np.random.default_rng(seed)
    target = min(count, len(keys))
    chosen = []
    quota = target // len(labels)
    for label in labels:
        pool = groups[label]
        if pool:
            chosen.extend((k, label) for k in rng.choice(pool, size=min(quota, len(pool)),
                                                        replace=False).tolist())
    chosen_ids = {key for key, _ in chosen}
    available = [(key, label) for label in labels for key in groups[label]
                 if key not in chosen_ids]
    rng.shuffle(available)
    chosen.extend(available[:target - len(chosen)])
    return sorted(chosen)


def summarize_rows(rows: list[dict], boundary_dropped: int,
                   failed_videos: list[dict], selected: list[tuple[str, str]]) -> dict:
    if not rows:
        raise ValueError("No eligible frames: inspect source ZIP and annotations")
    reasons = Counter(row["rejection_reason"] for row in rows)
    joints = Counter()
    by_label = {}
    by_stratum = {}
    accepted = reasons.get("accepted", 0)
    for row in rows:
        for joint in row["low_visibility_joints"]:
            joints[joint] += 1
    for target in ("error_knees_forward", "error_knees_inward"):
        by_label[target] = {}
        for value in (0, 1):
            part = [r for r in rows if r[target] == value]
            yes = sum(r["rejection_reason"] == "accepted" for r in part)
            by_label[target][str(value)] = {
                "eligible_frames": len(part), "accepted_frames": yes,
                "acceptance_fraction": (yes / len(part)) if part else None,
            }
    for stratum in sorted({x[1] for x in selected}):
        part = [r for r in rows if r["video_annotation_stratum"] == stratum]
        yes = sum(r["rejection_reason"] == "accepted" for r in part)
        by_stratum[stratum] = {
            "selected_videos": sum(label == stratum for _, label in selected),
            "eligible_frames": len(part), "accepted_frames": yes,
            "acceptance_fraction": yes / len(part) if part else None,
        }
    return {
        "selected_videos": len(selected),
        "videos_with_eligible_frames": len({r["video_id"] for r in rows}),
        "video_failures": failed_videos[:20],
        "video_failure_count": len(failed_videos),
        "annotation_boundary_dropped_frames": boundary_dropped,
        "eligible_frames": len(rows),
        "accepted_frames": accepted,
        "acceptance_fraction": accepted / len(rows),
        "rejection_reason_counts": {key: int(reasons.get(key, 0)) for key in REASONS},
        "low_visibility_joint_counts": dict(sorted(joints.items())),
        "by_error_annotation_at_timestamp": by_label,
        "by_video_annotation_stratum": by_stratum,
    }


def audit(archive_path: Path, annotations: Path, output_dir: Path, *,
          videos_per_split: int = 45, sample_fps: float = 2.0,
          seed: int = 42, min_visibility: float = .5,
          boundary_margin: float = .1, max_video_mb: int = 250,
          cv2_api=None, mp_api=None) -> dict:
    if videos_per_split < 1 or sample_fps <= 0 or not math.isfinite(sample_fps):
        raise ValueError("videos_per_split and FPS must be positive")
    if not (0 <= min_visibility <= 1 and 0 <= boundary_margin < 5 and max_video_mb > 0):
        raise ValueError("Invalid visibility, margin or maximum video size")
    source = {
        split: _read_json(annotations / f"{split}_keys.json")
        for split in ("train", "val")
    }
    if set(source["train"]) & set(source["val"]):
        raise ValueError("Publisher train/val video IDs overlap")
    forward = _read_json(annotations / "error_knees_forward.json")
    inward = _read_json(annotations / "error_knees_inward.json")
    if cv2_api is None or mp_api is None:
        from gym_pose.camera import _camera_libraries
        cv2_api, mp_api = _camera_libraries()
    picks = {
        split: stratified_video_sample(keys, forward, inward, count=videos_per_split,
                                      seed=seed + n)
        for n, (split, keys) in enumerate(source.items())
    }
    report = {
        "status": "pose_quality_DIAGNOSTIC_only_no_training_no_test",
        "dataset": "Fitness-AQA squat labeled",
        "selected_clip_sampling": "annotation_stratified_NOT_prevalence_representative",
        "random_seed": seed, "requested_videos_per_split": videos_per_split,
        "sample_fps": sample_fps, "boundary_margin_seconds": boundary_margin,
        "baseline_min_visibility": min_visibility,
        "alternative_visibility_cutoffs": list(VISIBILITY_THRESHOLDS),
        "splits": {},
        "warnings": [
            "Sample stratified by clip annotation; acceptance estimates NOT representative of full dataset",
            "Compare reason composition, not overall error prevalence",
            "A detected pose with low visibility is not necessarily inaccurate or accurate",
            "Relaxing visibility may increase coverage while degrading feature quality; DO NOT auto-change gate",
            "No real test-video access, frame images, model fitting or final evaluation",
            "Interval negative means no annotation of that named error, not perfect form",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = video_members(archive, {k for v in picks.values() for k, _ in v})
        with mp_api.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as estimator:
            for split, selected in picks.items():
                rows = []
                threshold_counters = Counter()
                failed = []
                boundaries = 0
                for clip_num, (key, stratum) in enumerate(selected, start=1):
                    entry = members[key]
                    try:
                        if entry.file_size > max_video_mb * 1024 * 1024:
                            raise ValueError("Video exceeds max-video-mb limit")
                        with tempfile.TemporaryDirectory(prefix="aqa_quality_") as directory:
                            tmp = Path(directory) / ("clip" + Path(entry.filename).suffix.lower())
                            with archive.open(entry) as src, tmp.open("wb") as dest:
                                shutil.copyfileobj(src, dest, length=1024 * 1024)
                            cap = cv2_api.VideoCapture(str(tmp))
                            try:
                                if not cap.isOpened():
                                    raise ValueError("OpenCV cannot open video")
                                fps = float(cap.get(cv2_api.CAP_PROP_FPS))
                                if fps <= 0 or not math.isfinite(fps):
                                    raise ValueError("Missing valid source video FPS")
                                frame_index, next_time = 0, 0.0
                                while True:
                                    ok, frame = cap.read()
                                    if not ok:
                                        break
                                    now = frame_index / fps
                                    idx = frame_index
                                    frame_index += 1
                                    if now + 1e-8 < next_time:
                                        continue
                                    next_time += 1.0 / sample_fps
                                    labels = labels_at_time(now, forward[key], inward[key],
                                                            margin=boundary_margin)
                                    if labels is None:
                                        boundaries += 1
                                        continue
                                    detection = estimator.process(
                                        cv2_api.cvtColor(frame, cv2_api.COLOR_BGR2RGB))
                                    landmarks = (detection.pose_landmarks.landmark
                                                 if detection.pose_landmarks else None)
                                    quality = assess_pose_quality(landmarks,
                                                                  min_visibility=min_visibility)
                                    for cutoff in VISIBILITY_THRESHOLDS:
                                        if assess_pose_quality(landmarks, min_visibility=cutoff)["reason"] == "accepted":
                                            threshold_counters[cutoff] += 1
                                    rows.append({
                                        "split": split, "video_id": key,
                                        "video_annotation_stratum": stratum,
                                        "frame_index": idx, "timestamp_s": round(now, 6),
                                        "error_knees_forward": labels[0],
                                        "error_knees_inward": labels[1],
                                        "rejection_reason": quality["reason"],
                                        "low_visibility_joints": quality["low_visibility_joints"],
                                        "minimum_critical_joint_visibility":
                                            quality["minimum_critical_joint_visibility"],
                                        "shoulder_distance": quality["shoulder_distance"],
                                    })
                            finally:
                                cap.release()
                    except (OSError, ValueError, zipfile.BadZipFile) as exc:
                        failed.append({"video_id": key, "reason": str(exc)})
                    if clip_num % 25 == 0:
                        print(f"{split}: audited {clip_num}/{len(selected)} clips", file=sys.stderr)
                summary = summarize_rows(rows, boundaries, failed, selected)
                summary["cutoff_accepted_counts_same_eligible_frames"] = {
                    str(th): {"accepted": int(threshold_counters[th]),
                              "acceptance_fraction": threshold_counters[th] / len(rows)}
                    for th in VISIBILITY_THRESHOLDS
                }
                report["splits"][split] = summary
                # Write no raw frames or source images. Frame-level metadata is
                # local and Git-ignored, useful for spot-checking issues.
                with (output_dir / f"quality_{split}_samples.csv").open(
                        "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    for row in rows:
                        data = row.copy()
                        data["low_visibility_joints"] = "|".join(data["low_visibility_joints"])
                        writer.writerow({k: data[k] for k in CSV_FIELDS})
    (output_dir / "pose_quality_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path,
                        default=Path("data/external/fitness_aqa/squat/labeled/videos.zip"))
    parser.add_argument("--annotations", type=Path,
                        default=Path("data/external/fitness_aqa/squat/labeled/annotations"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/fitness_aqa_pose_quality"))
    parser.add_argument("--videos-per-split", type=int, default=45)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-visibility", type=float, default=.5)
    parser.add_argument("--boundary-margin", type=float, default=.1)
    options = parser.parse_args()
    try:
        report = audit(
            options.zip, options.annotations, options.output_dir,
            videos_per_split=options.videos_per_split,
            sample_fps=options.sample_fps, seed=options.seed,
            min_visibility=options.min_visibility,
            boundary_margin=options.boundary_margin,
        )
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(1, f"Pose quality audit failed: {exc}\n")
    print(json.dumps({
        "status": report["status"],
        "splits": report["splits"],
        "full_report": str(options.output_dir / "pose_quality_report.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
