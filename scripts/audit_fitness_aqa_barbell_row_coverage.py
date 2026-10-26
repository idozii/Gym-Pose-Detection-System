#!/usr/bin/env python3
"""Source TRAIN/VAL Barbell Row MediaPipe visibility/pose-admission diagnostic.

The original Fitness-AQA Barbell Row labels are PER-IMAGE and each task
uses its own publisher train/val/test split. This tool only decodes selected
TRAIN/VAL images; test ZIP entries are *metadata only* and never decoded.
It does NOT train a model, re-extract features, select a deployment gate,
create image contact sheets, or overwrite any prior CSV/model artifact.

Alternative policies are hypothetical pose admissions, NOT validated
replacement feature vectors or improved classifier performance.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from gym_pose.barbell_row_features import (
    CRITICAL_JOINTS, TARGETS, pose_to_barbell_row_features,
)
from gym_pose.fitness_aqa_sources import (
    ROW_KEY, ROW_TASKS, inspect_barbell_row, load_splits, read_json,
)
from scripts.extract_fitness_aqa_barbell_row import _index_media, _pilot_selection

JOINT_NAMES = {
    11: "left_shoulder", 12: "right_shoulder",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}
TORSO4 = (11, 12, 23, 24)
SIX_NO_ANKLES = (11, 12, 23, 24, 25, 26)
POLICIES = {
    "all8_050_production": (CRITICAL_JOINTS, .50),
    "all8_035": (CRITICAL_JOINTS, .35),
    "all8_025": (CRITICAL_JOINTS, .25),
    "six_no_ankles_050": (SIX_NO_ANKLES, .50),
    "torso4_050": (TORSO4, .50),
    "torso4_035": (TORSO4, .35),
}
REASONS = (
    "accepted_by_current_32_features",
    "no_pose_detected",
    "invalid_required_joint_geometry",
    "critical_joint_low_visibility",
)


def assess_pose(landmarks) -> tuple[str, tuple[str, ...], dict[str, bool]]:
    """Check the current 32-feature gate and alternative admission policies.

    A 4-/6-joint policy is only an exploration of a DIFFERENT possible
    input schema; do not feed missing/unreliable coordinates to the existing
    32-feature classifier merely because a candidate policy accepts.
    """
    denied = {name: False for name in POLICIES}
    if landmarks is None:
        return "no_pose_detected", (), denied
    try:
        if len(landmarks) != 33:
            return "invalid_required_joint_geometry", (), denied
        xy = np.asarray(
            [[landmarks[i].x, landmarks[i].y, landmarks[i].z]
             for i in CRITICAL_JOINTS], dtype=float)
        vis = np.asarray(
            [float(landmarks[i].visibility) for i in CRITICAL_JOINTS],
            dtype=float)
    except (TypeError, ValueError, IndexError, AttributeError):
        return "invalid_required_joint_geometry", (), denied
    by_index = {
        key: (xy[pos], vis[pos]) for pos, key in enumerate(CRITICAL_JOINTS)
    }
    shoulder_width = np.linalg.norm(by_index[11][0] - by_index[12][0])
    if not np.isfinite(shoulder_width) or shoulder_width <= 1e-6:
        return "invalid_required_joint_geometry", (), denied

    policy_flags = {
        name: bool(all(
            np.isfinite(by_index[i][0]).all()
            and np.isfinite(by_index[i][1])
            and by_index[i][1] >= cutoff
            for i in indices
        )) for name, (indices, cutoff) in POLICIES.items()
    }
    low = tuple(
        JOINT_NAMES[i] for i in CRITICAL_JOINTS
        if np.isfinite(by_index[i][1]) and by_index[i][1] < .5
    )
    original = pose_to_barbell_row_features(landmarks, min_visibility=.5) is not None
    if policy_flags["all8_050_production"] != original:
        raise ValueError("Row baseline gate disagrees with extracted feature acceptance")
    if original:
        reason = "accepted_by_current_32_features"
    elif not (np.isfinite(xy).all() and np.isfinite(vis).all()):
        reason = "invalid_required_joint_geometry"
    else:
        reason = "critical_joint_low_visibility"
    return reason, low, policy_flags


def summarize(records: list[dict], *, selected: set[str], labels: dict) -> dict:
    """Report class-conditioned coverage on the SAME sampled IDs."""
    if not selected:
        raise ValueError("No original TRAIN/VAL images selected")
    inspected = {row["image_id"] for row in records}
    if len(inspected) != len(records) or not inspected <= selected:
        raise ValueError("Duplicate or non-publisher source-image diagnostic rows")
    failed = selected - inspected
    true_pos_selected = sum(labels[k] == 1 for k in selected)
    true_neg_selected = len(selected) - true_pos_selected
    reasons = Counter(r["reason"] for r in records)
    lower_vis = Counter(j for r in records for j in r["low_visibility_joints"])
    current = sum(r["gate_flags"]["all8_050_production"] for r in records)
    policies = {}
    for name in POLICIES:
        kept = [r for r in records if r["gate_flags"][name]]
        positive = sum(labels[r["image_id"]] == 1 for r in kept)
        negative = len(kept) - positive
        newly_kept = [
            r for r in kept if not r["gate_flags"]["all8_050_production"]
        ]
        policies[name] = {
            "accepted_images": len(kept),
            "accepted_fraction_of_selected": len(kept) / len(selected),
            "additional_images_vs_production": len(kept) - current,
            "positive_accepted": positive,
            "negative_accepted": negative,
            "positive_acceptance_rate": (
                positive / true_pos_selected if true_pos_selected else None),
            "negative_acceptance_rate": (
                negative / true_neg_selected if true_neg_selected else None),
            "new_positive_vs_production": sum(
                labels[r["image_id"]] == 1 for r in newly_kept),
            "new_negative_vs_production": sum(
                labels[r["image_id"]] == 0 for r in newly_kept),
        }
    return {
        "selected_images": len(selected),
        "selected_positive": true_pos_selected,
        "selected_negative": true_neg_selected,
        "decoded_and_pose_assessed_images": len(records),
        "image_decode_failures": len(failed),
        "failed_image_id_examples": sorted(failed)[:8],
        "reasons": {reason: reasons[reason] for reason in REASONS},
        "low_visibility_joint_occurrences": dict(sorted(lower_vis.items())),
        "candidate_admission_policies": policies,
    }


def audit(zip_path: Path, annotations: Path, output: Path, *,
          images_per_task_split: int = 180, seed: int = 42,
          max_image_mb: int = 24, max_pixels: int = 30_000_000,
          cv2_api=None, mp_api=None) -> dict:
    if (images_per_task_split < 1 or max_image_mb < 1 or max_pixels < 1
        or not isinstance(seed, int) or seed < 0):
        raise ValueError("Invalid audit sampling or image security limits")
    zip_path, annotations, output = map(Path, (zip_path, annotations, output))
    manifest = inspect_barbell_row(zip_path.parent)
    if not manifest["ready_for_task_separate_image_extraction"]:
        raise ValueError("Publisher Barbell Row source ZIP or metadata incomplete")
    if manifest["cross_task_split_audit"]["images_with_different_splits_between_tasks"]:
        raise ValueError("Publisher task split conflicts: cannot share decoder safely")
    splits = {
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
            task: set(_pilot_selection(
                splits[task][split], labels[task],
                max_images=images_per_task_split,
                seed=seed + 2 * split_idx + task_idx,
            ))
            for task_idx, task in enumerate(TARGETS)
        }
        for split_idx, split in enumerate(("train", "val"))
    }
    all_requested = set().union(*(
        selected[split][task] for split in ("train", "val") for task in TARGETS))
    if cv2_api is None or mp_api is None:
        from gym_pose.camera import _camera_libraries
        cv2_api, mp_api = _camera_libraries()
    report = {
        "status": "barbell_row_source_image_ANNOTATION_STRATIFIED_pose_gate_DIAGNOSTIC_NO_TEST",
        "exercise": "barbell_row", "original_media": "images",
        "seed": seed, "images_per_task_split": images_per_task_split,
        "original_publisher_test_images_decoded": 0,
        "separate_publisher_task_splits_preserved": True,
        "cross_task_exact_image_split_conflicts": 0,
        "candidate_admission_definitions": {
            name: {
                "critical_joints": [JOINT_NAMES[i] for i in joint_indices],
                "minimum_visibility": cutoff,
            } for name, (joint_indices, cutoff) in POLICIES.items()
        },
        "splits": {},
        "limitations": [
            "Samples are intentionally stratified by error label, NOT prevalence-representative",
            "Numbers are pose-admission counterfactuals, NOT AP/F1/accuracy or model quality",
            "Four- and six-joint policies need NEW feature schemas and training before use",
            "Relaxed visibility may admit inaccurate or physically unusable landmarks",
            "MediaPipe eight-joint features do NOT directly measure lumbar-spine curvature",
            "A 0 annotation means only no NAMED source error, not correct technique",
            "Publisher exact image-ID split integrity does not prove athlete/video independence",
            "TEST metadata only; original publisher TEST image contents were never decoded",
        ],
    }
    with zipfile.ZipFile(zip_path) as archive:
        media = _index_media(archive, all_requested)
        with mp_api.solutions.pose.Pose(
            static_image_mode=True, model_complexity=1) as pose:
            for split in ("train", "val"):
                ids = sorted(set().union(*selected[split].values()))
                observations = {}
                decode_failures = []
                for idx, key in enumerate(ids, 1):
                    entry = media[key]
                    try:
                        if entry.file_size > max_image_mb * 1024 * 1024:
                            raise ValueError("Image exceeds max_image_mb")
                        image = cv2_api.imdecode(
                            np.frombuffer(archive.read(entry), dtype=np.uint8),
                            cv2_api.IMREAD_COLOR)
                        if image is None or image.ndim != 3:
                            raise ValueError("Image decode failure")
                        if image.shape[0] * image.shape[1] > max_pixels:
                            raise ValueError("Decoded image exceeds max_pixels")
                        prediction = pose.process(
                            cv2_api.cvtColor(image, cv2_api.COLOR_BGR2RGB))
                        lm = (prediction.pose_landmarks.landmark
                              if prediction.pose_landmarks else None)
                        reason, low_vis, flags = assess_pose(lm)
                        observations[key] = {
                            "image_id": key, "reason": reason,
                            "low_visibility_joints": low_vis,
                            "gate_flags": flags,
                        }
                    except (OSError, TypeError, RuntimeError,
                            ValueError, zipfile.BadZipFile) as exc:
                        decode_failures.append({
                            "image_id": key, "cause": str(exc)
                        })
                    if idx % 200 == 0:
                        print(
                            f"{split}: inspected {idx}/{len(ids)} source images",
                            file=sys.stderr)
                by_task = {}
                for task in TARGETS:
                    task_rows = [
                        observations[key] for key in sorted(selected[split][task])
                        if key in observations
                    ]
                    by_task[task] = summarize(
                        task_rows, selected=selected[split][task],
                        labels=labels[task])
                report["splits"][split] = {
                    "unique_source_images_requested": len(ids),
                    "image_decode_or_assessment_failures": len(decode_failures),
                    "failure_examples": decode_failures[:8],
                    "tasks": by_task,
                }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path(
        "data/external/fitness_aqa/barbell_row/labeled/images.zip"))
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/external/fitness_aqa/barbell_row/labeled/annotations"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/fitness_aqa_barbell_row/pose_coverage_audit.json"))
    parser.add_argument("--images-per-task-split", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-image-mb", type=int, default=24)
    parser.add_argument("--max-pixels", type=int, default=30_000_000)
    args = parser.parse_args()
    try:
        report = audit(
            args.zip, args.annotations, args.output,
            images_per_task_split=args.images_per_task_split,
            seed=args.seed, max_image_mb=args.max_image_mb,
            max_pixels=args.max_pixels)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            ImportError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        parser.exit(1, f"Barbell Row gate audit failed: {exc}\n")
    print(json.dumps({
        "status": report["status"],
        "test_images_decoded": report["original_publisher_test_images_decoded"],
        "splits": report["splits"], "report": str(args.output),
    }, indent=2))
    return 2 if any(
        r["image_decode_or_assessment_failures"]
        for r in report["splits"].values()
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
