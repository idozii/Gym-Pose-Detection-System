#!/usr/bin/env python3
"""Publisher-aware dataset foundation inspector for all three Fitness-AQA exercises.

No image/video data is extracted, decoded or committed. Never trains a model.
For OHP and Barbell Row, validate original split metadata + labels and inspect
ZIP central directories. Squat reuses the proven squat-specific inspector.

By default, all 3 source datasets are checked (use --exercise for subsets).
With --skip-archive, checks metadata only and marks archive readiness unknown.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

from gym_pose.fitness_aqa_sources import inspect_barbell_row, inspect_ohp
from scripts.inspect_fitness_aqa import inspect as inspect_squat


def inspect(root: str | Path, *, exercises: tuple[str, ...] = (
            "squat", "ohp", "barbell_row"), skip_archive: bool = False) -> dict:
    root = Path(root)
    supported = ("squat", "ohp", "barbell_row")
    if not exercises or any(item not in supported for item in exercises):
        raise ValueError("Requested exercise must be squat, ohp or barbell_row")
    if len(set(exercises)) != len(exercises):
        raise ValueError("Duplicate exercises requested")
    result = {
        "milestone": "12",
        "dataset": "Fitness-AQA publisher exercise-specific labeled subsets",
        "dataset_home": "https://drive.google.com/drive/folders/16JWrJB-N7noUYNdZsZ_76suU_qArzzRn",
        "inspected_exercises": list(exercises),
        "archive_policy": "ZIP central-directory scan ONLY; no media extracted/decoded",
        "test_policy": "Official test IDs validated for integrity only; NO test model training/selection",
        "results": {},
        "all_requested_archives_ready": False,
        "warnings": [
            "Publisher splits are NOT verified athlete-identity disjoint",
            "Labels are named exercise-specific errors, not global correct/incorrect ground truth",
            "Cross-task Barbell Row publisher split conflicts MUST be audited before any shared multi-task model",
            "Squat media and error schemas differ from OHP and Barbell Row",
        ],
    }
    ready = []
    for exercise in exercises:
        if exercise == "squat":
            base = root / "squat" / "labeled"
            if skip_archive:
                # Old squat inspector is ZIP-coupled; reuse its publisher metadata
                # validator only when media is present. Do NOT claim ZIP checked.
                from gym_pose.fitness_aqa_sources import (
                    load_splits, read_json, validate_intervals, OHP_KEY,
                )
                splits = load_splits(base / "annotations", suffix="keys", key_pattern=OHP_KEY)
                assigned = set().union(*(set(v) for v in splits.values()))
                tasks = {
                    t: validate_intervals(
                        read_json(base / "annotations" / f"{t}.json"), assigned, task=t)
                    for t in ("error_knees_forward", "error_knees_inward")
                }
                summary = {
                    "exercise": "squat", "media_type": "video",
                    "annotation_semantics": "error_time_intervals_seconds",
                    "split_counts": {s: len(v) for s, v in splits.items()},
                    "assigned_unique_video_ids": len(assigned), "targets": tasks,
                    "archive": {"archive_checked": False, "archive_present": (base / "videos.zip").is_file(), "ready": False},
                    "ready_for_pose_extraction": False,
                }
            else:
                old = inspect_squat(base / "videos.zip", base / "annotations")
                checked = not (old["missing_video_keys_count"] or old["duplicate_video_stems_count"])
                summary = {
                    "exercise": "squat", "media_type": "video",
                    "annotation_semantics": "error_time_intervals_seconds",
                    "split_counts": old["split_counts"],
                    "assigned_unique_video_ids": old["total_unique_split_keys"],
                    "targets": old["annotation_video_counts"],
                    "archive": {**old, "archive_checked": True, "ready": checked},
                    "ready_for_pose_extraction": checked,
                }
            ready.append(summary["ready_for_pose_extraction"])
        elif exercise == "ohp":
            summary = inspect_ohp(root / "ohp" / "labeled", skip_archive=skip_archive)
            ready.append(summary["ready_for_pose_extraction"])
        else:
            summary = inspect_barbell_row(
                root / "barbell_row" / "labeled", skip_archive=skip_archive)
            ready.append(summary["ready_for_task_separate_image_extraction"])
        result["results"][exercise] = summary
    result["all_requested_archives_ready"] = bool(ready) and all(ready)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/external/fitness_aqa"))
    parser.add_argument("--exercise", choices=("all", "squat", "ohp", "barbell_row"), default="all")
    parser.add_argument("--skip-archive", action="store_true",
                        help="Validate label/split JSONs without requiring ZIPs")
    parser.add_argument("--output", type=Path, default=Path("artifacts/fitness_aqa_dataset/dataset_manifest.json"))
    args = parser.parse_args()
    exercises = ("squat", "ohp", "barbell_row") if args.exercise == "all" else (args.exercise,)
    try:
        report = inspect(args.root, exercises=exercises, skip_archive=args.skip_archive)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile,
            json.JSONDecodeError) as exc:
        print(f"Milestone 12 validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    print(f"Manifest: {args.output}", file=sys.stderr)
    if args.skip_archive:
        print("METADATA VALIDATED ONLY: ZIP presence/contents not checked", file=sys.stderr)
        return 0
    if not report["all_requested_archives_ready"]:
        print("SOURCE CHECK INCOMPLETE: missing or duplicate source media IDs", file=sys.stderr)
        return 2
    print("SOURCE ZIP + METADATA CHECK PASS (not training or test evaluation)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
