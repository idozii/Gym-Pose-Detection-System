"""M21: diagnose front-camera Push-up pose failures using TRAIN subjects ONLY.

This is a research diagnostic, NOT a new predictor or form-quality test.
Only previously rejected videos belonging to the M20 TRAIN subjects are
decoded. Validation and untouched TEST subjects are excluded automatically.
Only numeric per-frame reason counts are persisted, never image/video.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import zipfile

import numpy as np

from gym_pose.gym7_research import (
    frame_pose_vector, read_manifest, VIDEO_TYPES,
)
from gym_pose.gym7_training import load_clip_table, split_by_subject

PUSHUP_SIDES = (
    (11, 13, 15, 23, 25, 27),
    (12, 14, 16, 24, 26, 28),
)
JOINTS = {
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}
REASONS = (
    "no_pose", "invalid_landmarks", "low_one_side_visibility",
    "low_torso_scale", "accepted",
)


def diagnose_pushup_pose(landmarks, *, min_visibility: float = .35):
    """Match the current strict 132-feature acceptance rule exactly."""
    if landmarks is None or len(landmarks) != 33:
        return "no_pose", ()
    try:
        xyz = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=float)
        vis = np.asarray([p.visibility for p in landmarks], dtype=float)
    except (TypeError, AttributeError, ValueError):
        return "invalid_landmarks", ()
    if (xyz.shape != (33, 3) or vis.shape != (33,)
        or not np.isfinite(xyz).all() or not np.isfinite(vis).all()
        or np.any((vis < 0) | (vis > 1))):
        return "invalid_landmarks", ()
    if max(min(vis[list(side)]) for side in PUSHUP_SIDES) < min_visibility:
        absent = tuple(JOINTS[i] for i in JOINTS if vis[i] < min_visibility)
        return "low_one_side_visibility", absent
    # Mirrors M20's rejection scale: 2D distance between shoulder and hip centers.
    hip = (xyz[23] + xyz[24]) / 2
    shoulders = (xyz[11] + xyz[12]) / 2
    if float(np.linalg.norm(shoulders[:2] - hip[:2])) < .03:
        return "low_torso_scale", ()
    if frame_pose_vector(landmarks, "pushup", min_visibility=min_visibility) is None:
        return "invalid_landmarks", ()
    return "accepted", ()


def _inspect_one(entry, zipped, *, sample_fps: float, max_frames: int,
                 max_video_mb: int, cv2, mp) -> dict:
    if entry.file_size > max_video_mb * 1024 ** 2:
        return {"frames": 0, "accepted": 0, "reasons": {"oversize_video": 1},
                "missing_joints": {}}
    reasons = Counter()
    missing_joints = Counter()
    with tempfile.TemporaryDirectory(prefix="gym_pose_m21_diagnostic_") as folder:
        path = Path(folder) / ("clip" + PurePosixPath(entry.filename).suffix.lower())
        if path.suffix not in VIDEO_TYPES:
            raise ValueError("Unexpected video extension")
        with zipped.open(entry) as src, path.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                return {"frames": 0, "accepted": 0,
                        "reasons": {"unreadable_video": 1}, "missing_joints": {}}
            stride = max(1, int(round(fps / sample_fps)))
            frame_index = 0
            sampled = 0
            with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
                while sampled < max_frames:
                    ok, frame = capture.read()
                    if not ok: break
                    frame_index += 1
                    if (frame_index - 1) % stride: continue
                    sampled += 1
                    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    category, absent = diagnose_pushup_pose(
                        result.pose_landmarks.landmark if result.pose_landmarks else None
                    )
                    reasons[category] += 1
                    # Aggregate diagnostic counts only; never save frame pixels.
                    missing_joints.update(absent)
        finally:
            capture.release()
    return {
        "frames": sum(reasons.values()),
        "accepted": reasons.get("accepted", 0),
        "reasons": dict(reasons),
        "missing_joints": dict(missing_joints),
    }


def audit(archive: Path, manifest: Path, features: Path, report_path: Path, *,
          max_clips: int = 0, sample_fps: float = 2.,
          max_frames: int = 240, max_video_mb: int = 500) -> dict:
    if report_path.exists():
        raise FileExistsError("Existing M21 diagnostic report will not be overwritten")
    if not 0 < sample_fps <= 10 or max_clips < 0 or not 4 <= max_frames <= 2000:
        raise ValueError("Invalid diagnostic limits")
    if max_video_mb < 5:
        raise ValueError("Invalid maximum video size")

    source_rows = read_manifest(manifest)
    accepted = load_clip_table(features)
    splits = split_by_subject(accepted, seed=42)
    train_people = set(splits["train"].subject_id)
    non_train_people = set(splits["val"].subject_id) | set(splits["test"].subject_id)
    if train_people & non_train_people:
        raise AssertionError("Subject split leak")
    seen = set(accepted.video_id)
    eligible = sorted((
        r for r in source_rows
        if r["exercise"] == "pushup" and r["view"] == "front"
        and r["subject_id"] in train_people and r["video_id"] not in seen
    ), key=lambda r: r["video_id"])
    if max_clips:
        eligible = eligible[:max_clips]
    if not eligible:
        raise ValueError("No rejected TRAIN-only front Push-up videos found")

    # Fail closed if the source identity and persisted M20 metadata diverge.
    subject_map = {r["video_id"]: r["subject_id"] for r in source_rows}
    if any(subject_map.get(r.video_id) != r.subject_id for r in
           accepted.itertuples(index=False)):
        raise ValueError("M20 feature identities disagree with original manifest")

    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError("Install the M20 camera environment first") from error

    per_label = defaultdict(lambda: {
        "clips": 0, "clips_with_4_accepted_frames": 0,
        "sampled_frames": 0, "passed_frames": 0,
        "reasons": Counter(), "missing_joints": Counter()
    })
    with zipfile.ZipFile(archive) as zipped:
        members = {item.filename: item for item in zipped.infolist()}
        for row in eligible:
            entry = members.get(row["archive_member"])
            if entry is None or entry.is_dir():
                stats = {"frames": 0, "accepted": 0,
                         "reasons": {"missing_video": 1}, "missing_joints": {}}
            else:
                stats = _inspect_one(entry, zipped, sample_fps=sample_fps,
                    max_frames=max_frames, max_video_mb=max_video_mb, cv2=cv2, mp=mp)
            part = per_label[row["label"]]
            part["clips"] += 1
            part["clips_with_4_accepted_frames"] += int(stats["accepted"] >= 4)
            part["sampled_frames"] += stats["frames"]
            part["passed_frames"] += stats["accepted"]
            part["reasons"].update(stats["reasons"])
            part["missing_joints"].update(stats["missing_joints"])

    result = {
        "task": "M21 Push-up front-only M20 pose gate diagnosis",
        "source": "Mendeley Gym7 v3",
        "split": "TRAIN subjects from immutable M20 seed=42 split only",
        "validation_or_test_decoded": False,
        "threshold": .35,
        "minimum_accepted_frames_per_clip": 4,
        "selected_train_rejected_clips": len(eligible),
        "by_label": {
            label: {
                "clips": stats["clips"],
                "clips_with_4_accepted_frames": stats["clips_with_4_accepted_frames"],
                "sampled_frames": stats["sampled_frames"],
                "accepted_frames": stats["passed_frames"],
                "reasons": dict(stats["reasons"]),
                "missing_joints": dict(
                    sorted(stats["missing_joints"].items(),
                           key=lambda item: -item[1])),
            }
            for label, stats in sorted(per_label.items())
        },
        "note": (
            "Do not lower pose visibility thresholds before diagnosing causes. "
            "A diagnostic count does NOT validate quality labels or model accuracy."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path(
        "data/external/mendeley_gym7/Dataset Exercise-wise.zip"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/external/mendeley_gym7/m20_manifest.csv"))
    parser.add_argument("--features", type=Path, default=Path(
        "data/features/m20_gym7/m20_clip_features.csv"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/processed/m21_front_pose_audit.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        report = audit(args.zip, args.manifest, args.features, args.report,
                       max_clips=args.limit)
    except (OSError, ValueError, RuntimeError, ImportError, zipfile.BadZipFile) as error:
        print("M21 audit stopped: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
