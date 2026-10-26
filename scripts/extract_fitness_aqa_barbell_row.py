#!/usr/bin/env python3
"""Fitness-AQA Barbell Row *IMAGE* pose extraction (TRAIN and VAL only).

The publisher has two independently labeled 0/1 image tasks, each with its
own train/val/test image-ID split. Never decode official TEST images. The
source ZIP directory and publisher test keys may be audited for completeness
but test pixels are NEVER decoded, measured, or used to fit a model.

No temporal ordering, frame timing, clip or athlete identity is inferred
from filenames. Do not merge per-task split schemes or train a model here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from gym_pose.barbell_row_features import (
    FEATURE_NAMES, SCHEMA_VERSION, TARGETS, pose_to_barbell_row_features,
)
from gym_pose.barbell_row_torso4_features import (
    FEATURE_NAMES as TORSO4_FEATURE_NAMES,
    SCHEMA_VERSION as TORSO4_SCHEMA_VERSION,
    pose_to_torso4_features,
)
from gym_pose.fitness_aqa_sources import (
    IMAGE_EXTENSIONS, ROW_TASKS, inspect_barbell_row, load_splits,
    read_json, ROW_KEY,
)

META = ("image_id", "split", "target", "error_label", "candidate_clip_prefix")
COLUMNS = (*META, *FEATURE_NAMES)
PROFILE_NAMES = ("full8", "torso4")  # default preserves the legacy extractor



def _pilot_selection(ids: list[str], labels: dict, *, max_images: int, seed: int) -> list[str]:
    """Balance pilot source-image labels; report pilot as nonrepresentative."""
    keys = sorted(ids)
    if not max_images or len(keys) <= max_images:
        return keys
    positives = [k for k in keys if labels[k] == 1]
    negatives = [k for k in keys if labels[k] == 0]
    rng = np.random.default_rng(seed)
    chosen = []
    for group in (positives, negatives):
        want = min(len(group), max_images // 2)
        if want:
            chosen.extend(map(str, rng.choice(group, size=want, replace=False)))
    remaining = [k for k in keys if k not in set(chosen)]
    need = max_images - len(chosen)
    if need:
        chosen.extend(map(str, rng.choice(remaining, size=need, replace=False)))
    return sorted(chosen)


def _index_media(archive: zipfile.ZipFile, required_keys: set[str]):
    matched = {}
    for entry in archive.infolist():
        path = Path(entry.filename)
        if entry.is_dir() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem
        if key not in required_keys:
            continue
        if key in matched:
            raise ValueError(f"Duplicate requested image ID in ZIP: {key}")
        matched[key] = entry
    missing = required_keys - set(matched)
    if missing:
        raise ValueError(f"Missing publisher requested images in ZIP: {sorted(missing)[:8]}")
    return matched


def _prefix_overlap(assignments: dict[str, dict[str, str]]) -> dict:
    """Check source-ID prefix collisions across train/val WITHOUT assuming clip IDs."""
    stats = {}
    for task, mapping in assignments.items():
        a = {key.rsplit("_", 1)[0] for key, split in mapping.items() if split == "train"}
        b = {key.rsplit("_", 1)[0] for key, split in mapping.items() if split == "val"}
        stats[task] = {
            "candidate_prefix_overlap_train_val": len(a & b),
            "prefix_examples": sorted(a & b)[:8],
            "warning": "Filename prefix is NOT verified source-video/athlete ID",
        }
    return stats


def extract(archive_path: Path, annotations: Path, output_dir: Path, *,
            selected_splits: tuple[str, ...] = ("train", "val"),
            max_images_per_task_split: int = 12, seed: int = 42,
            min_visibility: float = .5, max_image_mb: int = 24,
            max_pixels: int = 30_000_000,
            cv2_api=None, mp_api=None, feature_profile: str = "full8") -> dict:
    if feature_profile not in PROFILE_NAMES:
        raise ValueError(f"Unknown Barbell Row feature profile: {feature_profile}")
    if (not selected_splits or len(set(selected_splits)) != len(selected_splits)
        or any(s not in ("train", "val") for s in selected_splits)):
        raise ValueError("Only source TRAIN and VAL images permitted; publisher TEST locked")
    if (max_images_per_task_split < 0 or max_image_mb < 1 or max_pixels < 1
        or not isinstance(seed, int) or seed < 0
        or not math.isfinite(min_visibility) or not 0 <= min_visibility <= 1):
        raise ValueError("Invalid Barbell Row image extraction parameters")
    archive_path, annotations, output_dir = map(
        Path, (archive_path, annotations, output_dir))
    source = inspect_barbell_row(archive_path.parent)
    if not source["ready_for_task_separate_image_extraction"]:
        raise ValueError("Barbell Row publisher media/label/split integrity failed")
    if source["cross_task_split_audit"]["images_with_different_splits_between_tasks"]:
        raise ValueError("Cross-task publisher split-ID conflicts; refuse shared decoding to prevent accidental leakage")
    split_map = {
        task: load_splits(annotations / "splits" / task,
                          suffix="ids", key_pattern=ROW_KEY)
        for task in TARGETS
    }
    labels = {
        task: read_json(annotations / filename)
        for task, filename in ROW_TASKS.items()
    }
    selected = {
        split: {
            task: _pilot_selection(
                split_map[task][split], labels[task],
                max_images=max_images_per_task_split,
                seed=seed + split_no * 2 + task_no,
            )
            for task_no, task in enumerate(TARGETS)
        }
        for split_no, split in enumerate(selected_splits)
    }
    # Need only source TRAIN+VAL requested image pixels; source TEST is never read.
    allowed = set().union(*(
        set(selected[split][task]) for split in selected_splits for task in TARGETS))
    if not allowed:
        raise ValueError("No source TRAIN/VAL images requested")
    assignments = {
        task: {
            key: split for split in selected_splits for key in selected[split][task]
        }
        for task in TARGETS
    }
    if feature_profile == "torso4":
        feature_names = TORSO4_FEATURE_NAMES
        schema_version = TORSO4_SCHEMA_VERSION
        extraction_function = pose_to_torso4_features
        report_basename = "fitness_aqa_barbell_row_torso4_extraction_summary.json"
        media_prefix = "fitness_aqa_barbell_row_torso4"
    else:
        feature_names = FEATURE_NAMES
        schema_version = SCHEMA_VERSION
        extraction_function = pose_to_barbell_row_features
        report_basename = "fitness_aqa_barbell_row_extraction_summary.json"
        media_prefix = "fitness_aqa_barbell_row"
    output_columns = (*META, *feature_names)
    if cv2_api is None or mp_api is None:
        from gym_pose.camera import _camera_libraries
        cv2_api, mp_api = _camera_libraries()

    summary = {
        "status": "barbell_row_image_pose_extraction_source_train_val_NO_TEST",
        "exercise": "barbell_row", "media_type": "image",
        "schema_version": schema_version, "feature_count": len(feature_names),
        "feature_profile": feature_profile,
        "feature_names": list(feature_names),
        "targets": list(TARGETS),
        "source_split_type": "publisher_TASK_SPECIFIC_image_ID_NOT_verified_subject_or_clip_disjoint",
        "source_test_images_decoded": 0,
        "max_images_per_task_split": max_images_per_task_split,
        "min_visibility": min_visibility, "seed": seed,
        "source_archive_extra_image_ids": source["archive"]["extra_keys_count"],
        "cross_target_split_id_conflicts": source["cross_task_split_audit"]["images_with_different_splits_between_tasks"],
        "candidate_clip_prefix_audit": _prefix_overlap(assignments),
        "splits": {},
        "warnings": [
            "Dataset labels are per-image binary 0/1, not OHP/Squat video time intervals",
            "A zero is only absence of the named publisher annotation, not globally correct form",
            "The two targets have DISTINCT publisher source splits; never naively unify for modeling",
            "The filename prefix is an unverified clip heuristic, not source video or athlete identity",
            "Lumbar-spine curvature is NOT directly measurable from these MediaPipe landmarks",
            "Torso4 features use shoulders and hips ONLY; not a medical spinal assessment",
            "No sequence, clip-level temporal model, training, performance scoring or TEST pixel access",
            "Pilot selection is label-stratified and not representative of publisher prevalence",
            "Pose abstentions are missing observations, never automatically class 0",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        media = _index_media(archive, allowed)
        with mp_api.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            for split in selected_splits:
                by_task = {
                    task: set(selected[split][task]) for task in TARGETS
                }
                ids = sorted(set().union(*by_task.values()))
                stats = {
                    "unique_images_to_decode": len(ids),
                    "images_decoded": 0, "image_decode_failures": 0,
                    "image_failure_examples": [],
                    "no_pose_images": 0,
                    "tasks": {
                        task: {
                            "selected_images": len(by_task[task]),
                            "pose_accepted": 0, "pose_rejected": 0,
                            "positive_accepted": 0, "negative_accepted": 0,
                            "positive_selected": sum(labels[task][k] for k in by_task[task]),
                        } for task in TARGETS
                    },
                }
                paths = {
                    task: output_dir / f"{media_prefix}_{task}_{split}_pose.csv"
                    for task in TARGETS
                }
                with ExitStack() as stack:
                    dest = {
                        task: stack.enter_context(paths[task].open(
                            "w", newline="", encoding="utf-8"))
                        for task in TARGETS
                    }
                    writers = {
                        task: csv.DictWriter(dest[task], fieldnames=output_columns)
                        for task in TARGETS
                    }
                    for writer in writers.values():
                        writer.writeheader()
                    for pos, key in enumerate(ids, start=1):
                        entry = media[key]
                        try:
                            if entry.file_size > max_image_mb * 1024 * 1024:
                                raise ValueError("Image exceeds input byte limit")
                            buf = archive.read(entry)
                            image = cv2_api.imdecode(
                                np.frombuffer(buf, dtype=np.uint8),
                                cv2_api.IMREAD_COLOR)
                            if image is None or image.ndim != 3:
                                raise ValueError("Cannot decode source image")
                            if image.shape[0] * image.shape[1] > max_pixels:
                                raise ValueError("Decoded source image exceeds max_pixels")
                            result = pose.process(cv2_api.cvtColor(
                                image, cv2_api.COLOR_BGR2RGB))
                            points = (result.pose_landmarks.landmark
                                      if result.pose_landmarks else None)
                            vector = (extraction_function(
                                points, min_visibility=min_visibility)
                                if points is not None else None)
                        except (OSError, ValueError, TypeError, zipfile.BadZipFile,
                                RuntimeError) as exc:
                            stats["image_decode_failures"] += 1
                            if len(stats["image_failure_examples"]) < 12:
                                stats["image_failure_examples"].append({
                                    "image_id": key, "reason": str(exc),
                                })
                            for task in TARGETS:
                                if key in by_task[task]:
                                    stats["tasks"][task]["pose_rejected"] += 1
                            continue
                        stats["images_decoded"] += 1
                        if points is None:
                            stats["no_pose_images"] += 1
                        for task in TARGETS:
                            if key not in by_task[task]:
                                continue
                            if vector is None:
                                stats["tasks"][task]["pose_rejected"] += 1
                                continue
                            label = labels[task][key]
                            writers[task].writerow({
                                "image_id": key, "split": split,
                                "target": task, "error_label": label,
                                "candidate_clip_prefix": key.rsplit("_", 1)[0],
                                **vector,
                            })
                            entry_counts = stats["tasks"][task]
                            entry_counts["pose_accepted"] += 1
                            entry_counts["positive_accepted" if label else "negative_accepted"] += 1
                        if pos % 500 == 0:
                            print(f"{split}: decoded {pos}/{len(ids)} source images",
                                  file=sys.stderr)
                for task in TARGETS:
                    row = stats["tasks"][task]
                    row["pose_coverage_selected"] = (
                        row["pose_accepted"] / row["selected_images"]
                        if row["selected_images"] else None)
                    row["csv"] = str(paths[task])
                summary["splits"][split] = stats

    report_path = output_dir / report_basename
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary_file"] = str(report_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path(
        "data/external/fitness_aqa/barbell_row/labeled/images.zip"))
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/external/fitness_aqa/barbell_row/labeled/annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/features/fitness_aqa_barbell_row"))
    parser.add_argument("--split", choices=("train", "val", "train-val"),
                        default="train-val")
    parser.add_argument("--max-images-per-task-split", type=int, default=12,
                        help="Pilot count per task/split; use 0 for complete publisher train/val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-visibility", type=float, default=.5)
    parser.add_argument("--max-image-mb", type=int, default=24)
    parser.add_argument("--max-pixels", type=int, default=30_000_000)
    args = parser.parse_args()
    splits = ("train", "val") if args.split == "train-val" else (args.split,)
    try:
        report = extract(
            args.zip, args.annotations, args.output_dir,
            selected_splits=splits,
            max_images_per_task_split=args.max_images_per_task_split,
            seed=args.seed, min_visibility=args.min_visibility,
            max_image_mb=args.max_image_mb, max_pixels=args.max_pixels)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            ImportError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        parser.exit(1, f"Barbell Row extraction failed: {exc}\n")
    print(json.dumps(report, indent=2))
    if any(s["image_decode_failures"] for s in report["splits"].values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
