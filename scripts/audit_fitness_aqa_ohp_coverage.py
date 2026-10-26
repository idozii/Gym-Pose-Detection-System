#!/usr/bin/env python3
"""Investigate low OHP pose coverage on stratified TRAIN and VAL clip samples.

Reads publisher source train/val clips from ZIP individually. No model fitting,
no TEST media access, no raw video/frame exports, no changes to existing CSVs.
Compare alternative visibility *admission rules on identical sampled frames*;
these are NOT validated alternative feature quality or model metrics.
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

from gym_pose.features import pose_to_features
from gym_pose.ohp_features import pose_to_ohp_features
from gym_pose.fitness_aqa_sources import inspect_ohp, read_json
from scripts.extract_fitness_aqa_pose import labels_at_time, video_members

JOINTS = {
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}
ALL = frozenset(JOINTS)
UPPER = frozenset((11, 12, 13, 14, 15, 16, 23, 24))
LOWER = frozenset((11, 12, 23, 24, 25, 26, 27, 28))
POLICIES = {
    "all12_050_production": (ALL, .5),
    "all12_035": (ALL, .35),
    "all12_025": (ALL, .25),
    "upper8_050_elbow_candidate": (UPPER, .5),
    "lower8_050_knee_candidate": (LOWER, .5),
}
REASONS = ("accepted", "no_pose_detected", "invalid_landmarks_or_geometry",
           "critical_joint_low_visibility")


def clip_stratum(elbow_windows, knee_windows) -> str:
    if elbow_windows and knee_windows:
        return "both_errors_annotated"
    if elbow_windows:
        return "elbows_annotated_only"
    if knee_windows:
        return "knees_annotated_only"
    return "neither_named_error_annotated"


def sample_clips(keys: list[str], elbows: dict, knees: dict, *, count: int, seed: int):
    if count < 1 or len(keys) != len(set(keys)):
        raise ValueError("Sample count must be positive and publisher clip IDs unique")
    groups = {name: [] for name in (
        "both_errors_annotated", "elbows_annotated_only",
        "knees_annotated_only", "neither_named_error_annotated")}
    for key in sorted(keys):
        if key not in elbows or key not in knees:
            raise ValueError(f"Annotation missing for source clip {key}")
        groups[clip_stratum(elbows[key], knees[key])].append(key)
    chosen = []
    rng = np.random.default_rng(seed)
    quota = min(count, len(keys)) // len(groups)
    for name, members in groups.items():
        if members:
            chosen.extend((str(k), name) for k in rng.choice(
                members, size=min(len(members), quota), replace=False))
    already = {key for key, _ in chosen}
    leftover = [(key, name) for name, members in groups.items()
                for key in members if key not in already]
    rng.shuffle(leftover)
    chosen.extend(leftover[:min(count, len(keys)) - len(chosen)])
    return sorted(chosen)


def classify(landmarks):
    """Return cause, low-vis joints and counterfactual gate membership.

    Returns a geometrically valid but low-visibility pose as a candidate,
    not as an evaluated reliable feature. If the production gate accepts it,
    exact agreement with pose_to_ohp_features is checked.
    """
    if landmarks is None:
        reason = "no_pose_detected"
        return reason, (), {name: False for name in POLICIES}
    try:
        if len(landmarks) != 33:
            raise ValueError("Unexpected landmark count")
        all_vis = np.asarray([float(p.visibility) for p in landmarks], dtype=float)
        if not np.isfinite(all_vis).all():
            raise ValueError("Nonfinite visibility")
        # The existing geometry function at zero cutoff verifies coordinates
        # and shoulder normalization without rejecting any nonnegative joints.
        if pose_to_features(landmarks, min_visibility=0.) is None:
            raise ValueError("Invalid pose geometry or coordinates")
    except (TypeError, ValueError, AttributeError, IndexError):
        return ("invalid_landmarks_or_geometry", (),
                {name: False for name in POLICIES})
    failing = tuple(JOINTS[i] for i in JOINTS if all_vis[i] < .5)
    flags = {
        policy: bool(all_vis[list(joints)].min() >= cutoff)
        for policy, (joints, cutoff) in POLICIES.items()
    }
    truly_accepted = pose_to_ohp_features(landmarks, min_visibility=.5) is not None
    if flags["all12_050_production"] != truly_accepted:
        raise ValueError("Production OHP quality gate disagrees with diagnostic gate")
    return ("accepted" if truly_accepted else "critical_joint_low_visibility",
            failing, flags)


def summarize(rows: list[dict], selected: list[tuple[str, str]],
              *, boundary_dropped: int, failed: list[dict]) -> dict:
    if not rows:
        raise ValueError("No valid annotation-timestamp sampled frames")
    reasons = Counter(row["reason"] for row in rows)
    joint_counts = Counter(j for row in rows for j in row["low_visibility_joints"])
    accepted = sum(r["gate_flags"]["all12_050_production"] for r in rows)
    policies = {}
    for policy in POLICIES:
        kept = [r for r in rows if r["gate_flags"][policy]]
        tasks = {}
        for target in ("error_elbows", "error_knees"):
            pos = sum(r[target] == 1 for r in rows)
            neg = len(rows) - pos
            kp = sum(r[target] == 1 for r in kept)
            kn = len(kept) - kp
            added = [r for r in kept if not r["gate_flags"]["all12_050_production"]]
            tasks[target] = {
                "eligible_positive_frames": pos,
                "accepted_positive_frames": kp,
                "new_positive_vs_production": sum(r[target] == 1 for r in added),
                "eligible_negative_frames": neg,
                "accepted_negative_frames": kn,
                "new_negative_vs_production": sum(r[target] == 0 for r in added),
                "positive_acceptance_fraction": (kp / pos) if pos else None,
                "negative_acceptance_fraction": (kn / neg) if neg else None,
            }
        policies[policy] = {
            "accepted_frames": len(kept),
            "accepted_fraction_of_eligible": len(kept) / len(rows),
            "additional_frames_vs_production": len(kept) - accepted,
            "clips_with_accepted_frames": len({r["video_id"] for r in kept}),
            "targets": tasks,
        }
    per_stratum = {}
    for stratum in sorted(set(name for _, name in selected)):
        part = [r for r in rows if r["stratum"] == stratum]
        per_stratum[stratum] = {
            "selected_clips": sum(name == stratum for _, name in selected),
            "eligible_frames": len(part),
            "production_accepted_frames": sum(r["gate_flags"]["all12_050_production"] for r in part),
        }
    return {
        "selected_clips": len(selected),
        "clips_with_eligible_frames": len({r["video_id"] for r in rows}),
        "failed_clip_count": len(failed), "failure_examples": failed[:12],
        "annotation_boundary_dropped_frames": boundary_dropped,
        "eligible_frames": len(rows),
        "reason_counts": {name: reasons[name] for name in REASONS},
        "low_visibility_joint_occurrences": dict(sorted(joint_counts.items())),
        "by_clip_annotation_stratum": per_stratum,
        "candidate_policies": policies,
    }


def audit(archive_path: Path, annotations: Path, output_path: Path, *,
          clips_per_split: int = 48, sample_fps: float = 2.,
          seed: int = 42, boundary_margin: float = .1,
          max_video_mb: int = 250, cv2_api=None, mp_api=None) -> dict:
    if (clips_per_split < 1 or max_video_mb < 1 or not math.isfinite(sample_fps)
        or sample_fps <= 0 or not math.isfinite(boundary_margin)
        or not 0 <= boundary_margin < 5):
        raise ValueError("Invalid sampling parameters")
    archive_path, annotations, output_path = map(
        Path, (archive_path, annotations, output_path))
    meta = inspect_ohp(archive_path.parent)
    if not meta["ready_for_pose_extraction"]:
        raise ValueError("Official source media IDs are not fully present")
    keys_by_split = {
        split: read_json(annotations / f"{split}_keys.json")
        for split in ("train", "val")
    }
    elbows = read_json(annotations / "error_elbows.json")
    knees = read_json(annotations / "error_knees.json")
    picks = {
        split: sample_clips(keys, elbows, knees, count=clips_per_split, seed=seed + n)
        for n, (split, keys) in enumerate(keys_by_split.items())
    }
    if cv2_api is None or mp_api is None:
        from gym_pose.camera import _camera_libraries
        cv2_api, mp_api = _camera_libraries()

    report = {
        "status": "ohp_source_train_val_stratified_pose_quality_DIAGNOSTIC_only",
        "sample_fps": sample_fps, "sampling_seed": seed,
        "requested_clips_per_split": clips_per_split,
        "annotation_boundary_margin_s": boundary_margin,
        "test_video_clips_accessed": 0,
        "candidate_policies": {name: {
            "critical_joint_names": [JOINTS[i] for i in sorted(indices)],
            "min_visibility": cutoff,
        } for name, (indices, cutoff) in POLICIES.items()},
        "splits": {},
        "limitations": [
            "Annotation-stratified pilot is NOT representative of the full source dataset",
            "These are pose-admission counterfactuals only, NOT model accuracy/AP/F1",
            "Relaxed or omitted joints can cause noisy/physically invalid feature estimates",
            "A named error-negative frame is NOT necessarily correct gym technique",
            "The original TEST split was not sampled or evaluated",
            "Publisher video-ID splits do not prove athlete/subject separation",
        ],
    }
    with zipfile.ZipFile(archive_path) as archive:
        members = video_members(
            archive, {k for picks_split in picks.values() for k, _ in picks_split})
        with mp_api.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            for split, selected in picks.items():
                rows, failures, boundaries = [], [], 0
                for idx, (key, stratum) in enumerate(selected, 1):
                    clip_rows = []
                    clip_boundaries = 0
                    try:
                        entry = members[key]
                        if entry.file_size > max_video_mb * 1024 * 1024:
                            raise ValueError("Video exceeds max_video_mb")
                        with tempfile.TemporaryDirectory(prefix="ohp_quality_") as folder:
                            tmp = Path(folder) / ("clip" + Path(entry.filename).suffix.lower())
                            with archive.open(entry) as source, tmp.open("wb") as dest:
                                shutil.copyfileobj(source, dest, length=1024 * 1024)
                            cap = cv2_api.VideoCapture(str(tmp))
                            try:
                                if not cap.isOpened():
                                    raise ValueError("Could not decode OHP video")
                                fps = float(cap.get(cv2_api.CAP_PROP_FPS))
                                if not math.isfinite(fps) or fps <= 0:
                                    raise ValueError("Source FPS unavailable")
                                frame_number, next_sample_s = 0, 0.
                                while True:
                                    ok, frame = cap.read()
                                    if not ok:
                                        break
                                    now = frame_number / fps
                                    frame_number += 1
                                    if now + 1e-8 < next_sample_s:
                                        continue
                                    next_sample_s += 1 / sample_fps
                                    labels = labels_at_time(
                                        now, elbows[key], knees[key], margin=boundary_margin)
                                    if labels is None:
                                        clip_boundaries += 1
                                        continue
                                    result = pose.process(
                                        cv2_api.cvtColor(frame, cv2_api.COLOR_BGR2RGB))
                                    lm = result.pose_landmarks.landmark if result.pose_landmarks else None
                                    reason, failing, gates = classify(lm)
                                    clip_rows.append({
                                        "video_id": key, "stratum": stratum,
                                        "reason": reason,
                                        "low_visibility_joints": failing,
                                        "gate_flags": gates,
                                        "error_elbows": labels[0],
                                        "error_knees": labels[1],
                                    })
                            finally:
                                cap.release()
                    except (OSError, ValueError, TypeError, zipfile.BadZipFile) as exc:
                        failures.append({"video_id": key, "reason": str(exc)})
                    else:
                        rows.extend(clip_rows)
                        boundaries += clip_boundaries
                    if idx % 25 == 0:
                        print(f"{split}: audited {idx}/{len(selected)} clips", file=sys.stderr)
                report["splits"][split] = summarize(
                    rows, selected, boundary_dropped=boundaries, failed=failures)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path(
        "data/external/fitness_aqa/ohp/labeled/videos.zip"))
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/external/fitness_aqa/ohp/labeled/annotations"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/fitness_aqa_ohp/pose_quality_audit.json"))
    parser.add_argument("--clips-per-split", type=int, default=48)
    parser.add_argument("--sample-fps", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-margin", type=float, default=.1)
    parser.add_argument("--max-video-mb", type=int, default=250)
    args = parser.parse_args()
    try:
        report = audit(
            args.zip, args.annotations, args.output,
            clips_per_split=args.clips_per_split, sample_fps=args.sample_fps,
            seed=args.seed, boundary_margin=args.boundary_margin,
            max_video_mb=args.max_video_mb)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            ImportError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"OHP pose quality audit failed: {exc}\n")
    print(json.dumps({
        "status": report["status"], "candidate_policies": report["candidate_policies"],
        "splits": report["splits"], "full_report": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
