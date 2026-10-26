"""Publisher metadata contracts for Fitness-AQA exercise-specific datasets.

Only source *clip / image* IDs are guaranteed from these files, NOT athlete or
session identity. This module deliberately makes no training/test selections.
"""
from __future__ import annotations

import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = frozenset((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"))
VIDEO_EXTENSIONS = frozenset((".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"))
OHP_KEY = re.compile(r"^[0-9]+_[0-9]+$")
ROW_KEY = re.compile(r"^[0-9]+_[0-9]+_[0-9]+$")
ROW_TASKS = {
    "torso_angle": "labels_torso_angle_error.json",
    "lumbar": "labels_lumbar_error.json",
}


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing publisher metadata: {path}")
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def load_splits(directory: Path, *, suffix: str, key_pattern: re.Pattern[str]) -> dict[str, list[str]]:
    splits = {}
    seen = {}
    for split in SPLITS:
        data = read_json(directory / f"{split}_{suffix}.json")
        if not isinstance(data, list) or any(
            not isinstance(k, str) or key_pattern.fullmatch(k) is None for k in data
        ):
            raise ValueError(f"{directory}/{split}: expected list of valid publisher image/video IDs")
        if len(data) != len(set(data)):
            raise ValueError(f"{directory}/{split}: duplicate source IDs within split")
        for key in data:
            if key in seen:
                raise ValueError(f"Data leakage: {key} appears in {seen[key]} and {split} splits")
            seen[key] = split
        splits[split] = data
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError(f"{directory}: all 3 publisher splits must contain IDs")
    return splits


def validate_intervals(labels: Any, keys: set[str], *, task: str) -> dict:
    if not isinstance(labels, dict):
        raise ValueError(f"{task}: expected video-ID -> interval-list mapping")
    missing = keys - set(labels)
    if missing:
        raise ValueError(f"{task}: missing annotation for {len(missing)} assigned videos: {sorted(missing)[:5]}")
    positives, intervals = 0, 0
    for key in keys:
        windows = labels[key]
        if not isinstance(windows, list):
            raise ValueError(f"{task}/{key}: expected list of time intervals")
        positives += bool(windows)
        prev_start = -1.0
        for pair in windows:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in pair)
                or pair[0] < 0 or pair[1] <= pair[0]):
                raise ValueError(f"{task}/{key}: invalid annotation interval {pair!r}")
            if pair[0] < prev_start:
                raise ValueError(f"{task}/{key}: intervals must be ordered by start time")
            prev_start = pair[0]
            intervals += 1
    return {"videos_with_annotated_error": int(positives), "annotated_intervals": intervals}


def validate_binary_labels(labels: Any, keys: set[str], *, task: str) -> dict:
    if not isinstance(labels, dict):
        raise ValueError(f"{task}: expected image-ID -> 0/1 mapping")
    missing = keys - set(labels)
    if missing:
        raise ValueError(f"{task}: missing label for {len(missing)} assigned images: {sorted(missing)[:5]}")
    values = [labels[k] for k in keys]
    if any(type(v) is not int or v not in (0, 1) for v in values):
        raise ValueError(f"{task}: labels must be integer 0 or 1 (not string, bool or missing)")
    return {"labeled_images": len(values), "positive_images": sum(values),
            "negative_images": len(values) - sum(values)}


def inspect_zip(path: Path | None, *, required_keys: set[str],
                extensions: frozenset[str], skip_archive: bool) -> dict:
    if skip_archive:
        return {
            "archive_checked": False, "archive_present": bool(path and path.is_file()),
            "media_files_found": None, "matched_keys": None, "missing_keys_count": None,
            "duplicates_count": None, "ready": False,
        }
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Required media ZIP missing: {path}; use --skip-archive for metadata-only validation")
    with zipfile.ZipFile(path) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        media = [entry for entry in entries if Path(entry.filename).suffix.lower() in extensions]
        stems = Counter(Path(entry.filename).stem for entry in media)
        dupes = sorted(k for k, count in stems.items() if count > 1)
        found = set(stems)
        missing = sorted(required_keys - found)
        extra = sorted(found - required_keys)
        report = {
            "archive_checked": True, "archive_present": True,
            "archive_path": str(path),
            "archive_uncompressed_bytes": sum(entry.file_size for entry in entries),
            "archive_total_entries": len(entries),
            "media_files_found": len(media),
            "media_extension_counts": dict(sorted(Counter(
                Path(e.filename).suffix.lower() for e in media).items())),
            "matched_keys": len(required_keys & found),
            "missing_keys_count": len(missing), "missing_key_examples": missing[:10],
            "extra_keys_count": len(extra), "extra_key_examples": extra[:10],
            "duplicates_count": len(dupes), "duplicate_key_examples": dupes[:10],
            "first_media_paths": [e.filename for e in media[:8]],
        }
        report["ready"] = bool(media) and not (missing or dupes)
        return report


def inspect_ohp(root: Path, *, skip_archive: bool = False) -> dict:
    """Validate *all* publisher OHP labels and source clips without decoding ZIP."""
    folder = root / "annotations"
    splits = load_splits(folder, suffix="keys", key_pattern=OHP_KEY)
    assigned = set().union(*(set(v) for v in splits.values()))
    tasks = {
        target: validate_intervals(read_json(folder / f"{target}.json"), assigned, task=target)
        for target in ("error_elbows", "error_knees")
    }
    archive = inspect_zip(
        root / "videos.zip", required_keys=assigned, extensions=VIDEO_EXTENSIONS,
        skip_archive=skip_archive,
    )
    return {
        "exercise": "ohp", "media_type": "video", "annotation_semantics": "error_time_intervals_seconds",
        "source_split_type": "publisher_video_ID_disjoint_NOT_verified_subject_disjoint",
        "split_counts": {s: len(keys) for s, keys in splits.items()},
        "assigned_unique_video_ids": len(assigned), "targets": tasks,
        "archive": archive,
        "ready_for_pose_extraction": archive["ready"],
        "warnings": [
            "Source split IDs do not verify athlete identity",
            "Absence of a named error interval is NOT a globally correct movement",
            "The OHP error labels and geometry are different from squat's labels",
            "Metadata inspection does not decode or validate source media content",
        ],
    }


def _collision_report(splits_by_task: dict[str, dict[str, list[str]]]) -> dict:
    """Detect cross-task split conflicts before any prospective multi-task model.

    Image IDs appearing in task-A train and task-B val/test cause leakage if
    one blindly concatenates them for a shared multi-task train/val/test split.
    Task-specific independent evaluations remain possible.
    """
    maps = {
        task: {key: split for split, keys in splits.items() for key in keys}
        for task, splits in splits_by_task.items()
    }
    a, b = maps["torso_angle"], maps["lumbar"]
    overlap = set(a) & set(b)
    conflicts = sorted(k for k in overlap if a[k] != b[k])
    counts = Counter(f"{a[k]}_vs_{b[k]}" for k in conflicts)
    return {
        "images_assigned_to_both_tasks": len(overlap),
        "images_with_different_splits_between_tasks": len(conflicts),
        "split_conflict_examples": [
            {"image_id": k, "torso_angle_split": a[k], "lumbar_split": b[k]}
            for k in conflicts[:8]
        ],
        "split_conflict_types": dict(sorted(counts.items())),
        "safe_to_naively_merge_into_joint_multitask_split": not conflicts,
    }


def _possible_clip_groups(splits: dict[str, list[str]]) -> dict:
    """Heuristic grouping only, not a verified video/subject identifier."""
    grouped = {name: {key.rsplit("_", 1)[0] for key in keys} for name, keys in splits.items()}
    conflicts = sorted((a, b, len(grouped[a] & grouped[b]))
                       for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:])
    return {
        "candidate_group_id_rule": "image_id.rsplit('_', 1)[0] (unverified prefix heuristic)",
        "prefix_overlap_counts": {f"{a}_vs_{b}": n for a, b, n in conflicts},
        "warning": "Prefixes are NOT verified clip IDs or athlete IDs; audit media provenance before grouped or subject-independent claims",
    }


def inspect_barbell_row(root: Path, *, skip_archive: bool = False) -> dict:
    """Validate task-specific image labels and splits, never silently unify them."""
    base = root / "annotations"
    by_task = {}
    splits_by_task = {}
    all_assigned = set()
    for task, filename in ROW_TASKS.items():
        splits = load_splits(base / "splits" / task, suffix="ids", key_pattern=ROW_KEY)
        assigned = set().union(*(set(v) for v in splits.values()))
        labels = read_json(base / filename)
        counts = validate_binary_labels(labels, assigned, task=task)
        by_task[task] = {
            "split_counts": {split: len(keys) for split, keys in splits.items()},
            "assigned_unique_image_ids": len(assigned),
            "targets": {task: counts},
            "possible_clip_prefix_overlap": _possible_clip_groups(splits),
        }
        splits_by_task[task] = splits
        all_assigned |= assigned
    conflicts = _collision_report(splits_by_task)
    archive = inspect_zip(
        root / "images.zip", required_keys=all_assigned,
        extensions=IMAGE_EXTENSIONS, skip_archive=skip_archive,
    )
    return {
        "exercise": "barbell_row", "media_type": "image",
        "annotation_semantics": "per_image_binary_error_label",
        "source_split_type": "publisher_task_specific_image_ID_splits_NOT_verified_subject_disjoint",
        "tasks": by_task,
        "unique_images_required_across_tasks": len(all_assigned),
        "cross_task_split_audit": conflicts, "archive": archive,
        "ready_for_task_separate_image_extraction": archive["ready"],
        "ready_for_unified_multitask_training": archive["ready"] and (
            conflicts["safe_to_naively_merge_into_joint_multitask_split"]
            and all(v == 0 for t in by_task.values() for v in
                    t["possible_clip_prefix_overlap"]["prefix_overlap_counts"].values())
        ),
        "warnings": [
            "Torso-angle and lumbar have DISTINCT publisher splits; do not combine naively",
            "Labels are per-image binary, NOT time intervals, and do not prove globally correct form",
            "Image-ID prefix may reflect a source clip; actual group and athlete identity are unverified",
            "Any cross-task split conflicts prevent naive shared model training",
            "Even when IDs are disjoint, near-duplicate image and subject overlap can cause leakage",
            "The squat temporal 119-feature sequence model cannot be reused blindly on independent images",
        ],
    }
