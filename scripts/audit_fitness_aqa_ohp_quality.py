#!/usr/bin/env python3
"""Diagnose OHP pose exclusion on publisher TRAIN/VAL sampled source clips.

Never inspect publisher TEST media, fit models, change gate thresholds, or
rewrite the previous OHP extraction. Purposely annotation-stratified videos:
the pilot's aggregate fractions are NOT prevalence-representative.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from gym_pose.fitness_aqa_sources import inspect_ohp, read_json
from gym_pose.ohp_features import pose_to_ohp_features
from gym_pose.ohp_quality import (
    REASONS, TASK_GATE_POLICIES, assess_ohp_pose_quality, assess_task_gate_policies,
)
from scripts.extract_fitness_aqa_pose import labels_at_time, video_members

TARGETS = ("error_elbows", "error_knees")
STRATA = ("knees_annotated_video", "elbows_only_annotated_video",
          "neither_error_annotated_video")


def stratum(elbows: list, knees: list) -> str:
    if knees:
        return STRATA[0]
    if elbows:
        return STRATA[1]
    return STRATA[2]


def select_videos(keys: list[str], elbows: dict, knees: dict, *,
                  count: int, seed: int) -> list[tuple[str, str]]:
    """Sample equal initial quota across three strata, then fill remainder.

    Keep sample-video choice independent from MediaPipe outcomes and avoid
    cherry-picking clips based on observed model/pose performance.
    """
    if count < 1 or seed < 0:
        raise ValueError("count must be positive, seed nonnegative")
    if len(set(keys)) != len(keys) or any(k not in elbows or k not in knees for k in keys):
        raise ValueError("Duplicate or missing labeled OHP video IDs")
    groups = {group: [] for group in STRATA}
    for key in sorted(keys):
        groups[stratum(elbows[key], knees[key])].append(key)
    rng = np.random.default_rng(seed)
    target = min(count, len(keys))
    selection = []
    quota = target // len(STRATA)
    for group in STRATA:
        pool = groups[group]
        if pool:
            selection.extend(
                (str(key), group)
                for key in rng.choice(pool, size=min(quota, len(pool)), replace=False)
            )
    taken = {key for key, _ in selection}
    remaining = [(key, group) for group in STRATA for key in groups[group] if key not in taken]
    rng.shuffle(remaining)
    selection.extend(remaining[:target - len(selection)])
    return sorted(selection)


def inspect_clip(entry: zipfile.ZipInfo, archive: zipfile.ZipFile,
                 cv2_api, pose, *, key: str, split: str, category: str,
                 elbows: list, knees: list, sample_fps: float,
                 boundary_margin: float, baseline_visibility: float,
                 diagnostic_visibility: float, max_video_bytes: int) -> tuple[list[dict], int]:
    if entry.file_size > max_video_bytes:
        raise ValueError(f"{key}: archive member exceeds max-video-mb")
    records, boundary = [], 0
    with tempfile.TemporaryDirectory(prefix="fitness_aqa_ohp_audit_") as folder:
        path = Path(folder) / ("clip" + Path(entry.filename).suffix.lower())
        with archive.open(entry) as source, path.open("wb") as dest:
            shutil.copyfileobj(source, dest, length=1024 * 1024)
        reader = cv2_api.VideoCapture(str(path))
        try:
            if not reader.isOpened():
                raise ValueError(f"OpenCV cannot decode {key}")
            fps = float(reader.get(cv2_api.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"Invalid source FPS for {key}: {fps}")
            index, next_sample = 0, 0.0
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                timestamp = index / fps
                current_index = index
                index += 1
                if timestamp + 1e-8 < next_sample:
                    continue
                next_sample += 1.0 / sample_fps
                annotation = labels_at_time(
                    timestamp, elbows, knees, margin=boundary_margin)
                if annotation is None:
                    boundary += 1
                    continue
                detected = pose.process(cv2_api.cvtColor(frame, cv2_api.COLOR_BGR2RGB))
                landmarks = (
                    detected.pose_landmarks.landmark
                    if getattr(detected, "pose_landmarks", None) else None
                )
                quality = assess_ohp_pose_quality(
                    landmarks, min_visibility=baseline_visibility)
                existing_accept = (pose_to_ohp_features(
                    landmarks, min_visibility=baseline_visibility) is not None
                    if landmarks is not None else False)
                if existing_accept != (quality["reason"] == "accepted"):
                    raise AssertionError(f"{key}: quality audit differs from production gate")
                relaxed_accept = (
                    pose_to_ohp_features(landmarks, min_visibility=diagnostic_visibility)
                    is not None if landmarks is not None else False
                )
                if existing_accept and not relaxed_accept:
                    raise AssertionError("Lowered visibility threshold rejected an accepted pose")
                gates = assess_task_gate_policies(
                    landmarks, strict=baseline_visibility,
                    relaxed=diagnostic_visibility,
                )
                if (gates["all_050"] != existing_accept
                    or gates["all_035"] != relaxed_accept):
                    raise AssertionError("Counterfactual gates disagree with production OHP gate")
                records.append({
                    "video_id": key, "split": split,
                    "video_annotation_stratum": category,
                    "frame_index": current_index,
                    "timestamp_s": round(timestamp, 6),
                    "error_elbows": annotation[0],
                    "error_knees": annotation[1],
                    "rejection_reason": quality["reason"],
                    "low_visibility_joints": quality["low_visibility_joints"],
                    "low_visibility_base_joints": quality["low_visibility_base_joints"],
                    "low_visibility_upper_joints": quality["low_visibility_upper_joints"],
                    "accepted_at_baseline": existing_accept,
                    "accepted_at_diagnostic_cutoff": relaxed_accept,
                    "visibility_policies": gates,
                })
        finally:
            reader.release()
    return records, boundary


def summarize(rows: list[dict], picked: list[tuple[str,str]], *,
              rejected_boundary: int, failures: list[dict],
              diagnostic_visibility: float) -> dict:
    if not rows:
        raise ValueError("No eligible pose frames in pilot; inspect video decode/annotation timing")
    reasons = Counter(r["rejection_reason"] for r in rows)
    joints = Counter(name for r in rows for name in r["low_visibility_joints"])
    baseline = sum(r["accepted_at_baseline"] for r in rows)
    relaxed = sum(r["accepted_at_diagnostic_cutoff"] for r in rows)
    by_error = {}
    for target in TARGETS:
        by_error[target] = {}
        for val in (0,1):
            subset = [r for r in rows if r[target] == val]
            current = sum(r["accepted_at_baseline"] for r in subset)
            alternate = sum(r["accepted_at_diagnostic_cutoff"] for r in subset)
            by_error[target][str(val)] = {
                "eligible_frames": len(subset),
                "accepted_baseline": current,
                "accepted_alternative": alternate,
                "newly_accepted_alternative": alternate - current,
                "baseline_coverage": current / len(subset) if subset else None,
            }
    # Note: alternative gates are *diagnostic admission rules only*.
    # No OHP model supports those masks at this time.
    task_gate_counts = {}
    for policy in TASK_GATE_POLICIES:
        count = sum(row["visibility_policies"][policy] for row in rows)
        task_gate_counts[policy] = {
            "accepted_frames": count,
            "coverage": count / len(rows),
            "newly_accepted_vs_all_050": count - baseline,
            "newly_admitted_positive_annotations": {
                task: int(sum(
                    row[task] == 1 and row["visibility_policies"][policy]
                    and not row["accepted_at_baseline"]
                    for row in rows
                )) for task in TARGETS
            },
        }
    if task_gate_counts["all_050"]["accepted_frames"] != baseline:
        raise AssertionError("Recomputed strict gate count differs")
    if task_gate_counts["all_035"]["accepted_frames"] != relaxed:
        raise AssertionError("Recomputed relaxed gate count differs")
    task_specific = {
        task: {
            "candidate_policy": policy,
            "eligible_positive_frames": int(sum(row[task] == 1 for row in rows)),
            "accepted_positive_baseline": int(sum(
                row[task] == 1 and row["accepted_at_baseline"] for row in rows
            )),
            "accepted_positive_candidate": int(sum(
                row[task] == 1 and row["visibility_policies"][policy] for row in rows
            )),
            "accepted_negative_baseline": int(sum(
                row[task] == 0 and row["accepted_at_baseline"] for row in rows
            )),
            "accepted_negative_candidate": int(sum(
                row[task] == 0 and row["visibility_policies"][policy] for row in rows
            )),
        } for task, policy in (
            ("error_elbows", "elbows_arm_chain_050"),
            ("error_knees", "knees_lower_chain_050"),
        )
    }
    by_stratum = {}
    for category in STRATA:
        subset = [r for r in rows if r["video_annotation_stratum"] == category]
        by_stratum[category] = {
            "selected_video_count": sum(cat == category for _,cat in picked),
            "eligible_frames": len(subset),
            "accepted_baseline": sum(r["accepted_at_baseline"] for r in subset),
        }
    return {
        "selected_videos": len(picked),
        "processed_videos_with_eligible_frames": len({r["video_id"] for r in rows}),
        "video_failures": failures[:20], "videos_failed": len(failures),
        "annotation_boundary_rejected_frames": rejected_boundary,
        "eligible_frames": len(rows), "accepted_baseline_frames": baseline,
        "baseline_coverage": baseline / len(rows),
        "candidate_cutoff": diagnostic_visibility,
        "accepted_alternative_frames": relaxed,
        "alternative_coverage": relaxed / len(rows),
        "additional_candidate_frames": relaxed - baseline,
        "rejection_reason_counts": {k:int(reasons.get(k,0)) for k in REASONS},
        "low_visibility_joint_counts": dict(sorted(joints.items())),
        "task_visibility_admission_policies": task_gate_counts,
        "task_specific_positive_recovery": task_specific,
        "by_error_annotation_at_timestamp": by_error,
        "by_publisher_video_annotation_stratum": by_stratum,
    }


def audit(archive_path: Path, annotation_dir: Path, output: Path, *,
          videos_per_split: int = 30, sample_fps: float = 2.0,
          boundary_margin: float = .1, baseline_visibility: float = .5,
          diagnostic_visibility: float = .35, seed: int = 42,
          max_video_mb: int = 250, cv2_api=None, mp_api=None) -> dict:
    if (videos_per_split < 1 or seed < 0 or max_video_mb < 1
        or sample_fps <= 0 or not math.isfinite(sample_fps)
        or not math.isfinite(boundary_margin) or not 0 <= boundary_margin < 5
        or not 0 <= diagnostic_visibility <= baseline_visibility <= 1):
        raise ValueError("Invalid sampling, visibility, seed or max size")
    base = Path(archive_path)
    annotation_dir = Path(annotation_dir)
    source = inspect_ohp(base.parent)
    if not source["ready_for_pose_extraction"]:
        raise ValueError("OHP archive integrity check did not pass")
    ids = {split: read_json(annotation_dir / f"{split}_keys.json")
           for split in ("train","val")}
    elbows = read_json(annotation_dir / "error_elbows.json")
    knees = read_json(annotation_dir / "error_knees.json")
    picked = {
        split: select_videos(ids[split], elbows, knees, count=videos_per_split,
                             seed=seed + offset)
        for offset,split in enumerate(("train","val"))
    }
    if cv2_api is None or mp_api is None:
        from gym_pose.camera import _camera_libraries
        cv2_api,mp_api = _camera_libraries()
    result = {
        "status": "OHP_train_val_annotation_stratified_pose_quality_DIAGNOSTIC_ONLY",
        "seed": seed, "sample_fps": sample_fps,
        "boundary_margin_seconds": boundary_margin,
        "baseline_visibility": baseline_visibility,
        "alternative_visibility_for_DIAGNOSTIC_only": diagnostic_visibility,
        "source_test_clips_decoded": 0,
        "data_selection": "publisher TRAIN/VAL video IDs with balanced annotation strata",
        "splits": {},
        "warnings": [
            "Pilot sample deliberately annotation-stratified, not population representative",
            "Lower gate acceptance may yield noisy pose features; never auto-deploy this cutoff",
            "A no-error annotation only means the named error was not tagged at that timestamp",
            "Frames are correlated within videos and original athlete IDs are unverified",
            "TEST media was not decoded, no model trained or evaluated, no feature CSV overwritten",
            "Task-specific visibility policies are hypothetical frame-admission criteria, NOT production 111-feature model inputs",
            "Knee-error classification still needs reliable knee and ankle geometry; elbow classification needs visible arm joints",
            "Source train/val exploratory clips were sampled by annotation stratum; these counts are NOT population-weighted",
        ],
    }
    with zipfile.ZipFile(base) as archive:
        allowed = set().union(*(set(key for key,_ in picks) for picks in picked.values()))
        members = video_members(archive, allowed)
        with mp_api.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as estimator:
            for split,selection in picked.items():
                rows,failed, boundaries = [], [], 0
                for idx,(key,category) in enumerate(selection, 1):
                    try:
                        clip_rows,dropped = inspect_clip(
                            members[key], archive, cv2_api, estimator,
                            key=key, split=split, category=category,
                            elbows=elbows[key], knees=knees[key],
                            sample_fps=sample_fps, boundary_margin=boundary_margin,
                            baseline_visibility=baseline_visibility,
                            diagnostic_visibility=diagnostic_visibility,
                            max_video_bytes=max_video_mb * 1024 * 1024,
                        )
                        rows.extend(clip_rows)
                        boundaries += dropped
                    except (OSError, ValueError, AssertionError, zipfile.BadZipFile) as exc:
                        failed.append({"video_id":key,"error":str(exc)})
                    if idx % 20 == 0:
                        print(f"{split}: audited {idx}/{len(selection)} videos",
                              file=sys.stderr)
                result["splits"][split] = summarize(
                    rows,selection,rejected_boundary=boundaries,
                    failures=failed,diagnostic_visibility=diagnostic_visibility)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path(
        "data/external/fitness_aqa/ohp/labeled/videos.zip"))
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/external/fitness_aqa/ohp/labeled/annotations"))
    parser.add_argument("--videos-per-split", type=int, default=30)
    parser.add_argument("--sample-fps", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-visibility", type=float, default=.5)
    parser.add_argument("--alternative-visibility", type=float, default=.35)
    parser.add_argument("--max-video-mb", type=int, default=250)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/fitness_aqa_ohp_quality/diagnostic.json"))
    args = parser.parse_args()
    try:
        result = audit(args.zip, args.annotations,args.output,
                       videos_per_split=args.videos_per_split,
                       sample_fps=args.sample_fps, seed=args.seed,
                       baseline_visibility=args.min_visibility,
                       diagnostic_visibility=args.alternative_visibility,
                       max_video_mb=args.max_video_mb)
    except (OSError, ValueError, TypeError, KeyError, AssertionError,
            RuntimeError, ImportError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"OHP quality audit failed: {exc}\n")
    print(json.dumps({"status": result["status"],
                      "splits": result["splits"],
                      "full_report": str(args.output)},indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
