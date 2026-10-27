"""M22 - paired TRAIN-only Push-up front-camera pose admission ablation.

Research diagnostic only. Compares original full-body vs upper-body gates
under baseline and alternative MediaPipe detectors on identical source frames.
NEVER uses validation or TEST participant videos, scores, or model weights.
No frame/image, landmark coordinates, video ID or participant ID is persisted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import zipfile

import numpy as np

from gym_pose.gym7_research import read_manifest, VIDEO_TYPES
from gym_pose.gym7_training import load_clip_table, split_by_subject
from scripts.m21_audit_front_pose import diagnose_pushup_pose

UPPER_SIDES = ((11, 13, 15, 23), (12, 14, 16, 24))
MIN_VISIBILITY = .35
MIN_GOOD_FRAMES = 4
DETECTORS = {
    "baseline": {"model_complexity": 1, "min_detection_confidence": .5},
    "heavy": {"model_complexity": 2, "min_detection_confidence": .5},
    "lower_detection_threshold": {
        "model_complexity": 1, "min_detection_confidence": .35
    },
}
GATES = ("full_body", "upper_body")
STATES = (
    "no_pose", "invalid_landmarks", "low_one_side_visibility",
    "low_upper_body_visibility", "low_torso_scale", "accepted"
)


def upper_body_gate(landmarks, *, min_visibility: float = MIN_VISIBILITY) -> str:
    """Only a TRAIN-only diagnostic, NOT a production quality feature schema."""
    if landmarks is None or len(landmarks) != 33:
        return "no_pose"
    try:
        xyz = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=float)
        vis = np.asarray([p.visibility for p in landmarks], dtype=float)
    except (TypeError, ValueError, AttributeError):
        return "invalid_landmarks"
    if (xyz.shape != (33, 3) or vis.shape != (33,)
        or not np.isfinite(xyz).all() or not np.isfinite(vis).all()
        or np.any(vis < 0) or np.any(vis > 1)):
        return "invalid_landmarks"
    if max(min(vis[list(side)]) for side in UPPER_SIDES) < min_visibility:
        return "low_upper_body_visibility"
    hip = (xyz[23] + xyz[24]) / 2
    shoulder = (xyz[11] + xyz[12]) / 2
    if float(np.linalg.norm(shoulder[:2] - hip[:2])) < .03:
        return "low_torso_scale"
    return "accepted"


def select_train_rejected_front(manifest_rows, extracted, *, seed: int = 42):
    """Keep fixed M20 participant split, don't decode validation or TEST media.

    Reading feature metadata for all subjects is necessary to RECONSTRUCT
    the original split. Neither val/test videos nor their predictions are used.
    """
    parts = split_by_subject(extracted, seed=seed)
    train_subjects = set(parts["train"].subject_id)
    others = set(parts["val"].subject_id) | set(parts["test"].subject_id)
    if train_subjects & others:
        raise AssertionError("TRAIN/holdout subject overlap")
    source = {r["video_id"]: r for r in manifest_rows}
    if len(source) != len(manifest_rows):
        raise ValueError("Duplicate source video_id")
    for sample in extracted.itertuples(index=False):
        entry = source.get(sample.video_id)
        if entry is None or entry["subject_id"] != sample.subject_id:
            raise ValueError("M20 feature identities differ from source manifest")
    accepted_ids = set(extracted.video_id)
    selected = [
        r for r in manifest_rows
        if r["subject_id"] in train_subjects and
           r["exercise"] == "pushup" and r["view"] == "front" and
           r["video_id"] not in accepted_ids
    ]
    return sorted(selected, key=lambda r: r["video_id"])


def _inspect_video(member, archive, models, cv2, *, sample_fps,
                   max_frames, max_video_mb):
    if member is None or member.is_dir():
        return {"file_error": "missing_archive_video"}
    if member.file_size > max_video_mb * 1024 ** 2:
        return {"file_error": "oversize_video"}
    if PurePosixPath(member.filename).suffix.lower() not in VIDEO_TYPES:
        return {"file_error": "invalid_extension"}

    counters = {
        model: {gate: Counter() for gate in GATES}
        for model in models
    }
    sampled = 0
    with tempfile.TemporaryDirectory(prefix="m22_pose_audit_") as temp:
        video = Path(temp) / ("clip" + PurePosixPath(member.filename).suffix.lower())
        with archive.open(member) as inp, video.open("xb") as out:
            shutil.copyfileobj(inp, out, length=1024 * 1024)
        capture = cv2.VideoCapture(str(video))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                return {"file_error": "unreadable_video"}
            stride = max(1, round(fps / sample_fps))
            frame_index = 0
            while sampled < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                if (frame_index - 1) % stride:
                    continue
                sampled += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for mode, detector in models.items():
                    output = detector.process(rgb)
                    landmark = (
                        output.pose_landmarks.landmark
                        if output.pose_landmarks else None
                    )
                    full_reason, _ = diagnose_pushup_pose(landmark)
                    upper_reason = upper_body_gate(landmark)
                    counters[mode]["full_body"][full_reason] += 1
                    counters[mode]["upper_body"][upper_reason] += 1
        finally:
            capture.release()
    return {"sampled": sampled, "modes": counters}


def audit(archive: Path, manifest: Path, features: Path, output: Path, *,
          seed: int = 42, limit: int = 0, sample_fps: float = 2.,
          max_frames: int = 240, max_video_mb: int = 500):
    if output.exists():
        raise FileExistsError("M22 output already exists - do not overwrite")
    if (seed != 42 or limit < 0 or not 0 < sample_fps <= 10
        or not 4 <= max_frames <= 2000 or max_video_mb < 5):
        raise ValueError("Invalid limits or non-original M20 split seed")
    rows = read_manifest(manifest)
    extracted = load_clip_table(features)
    selected = select_train_rejected_front(rows, extracted, seed=seed)
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No TRAIN rejected front Push-up videos available")

    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError("Use the isolated .venv-m20 environment") from error

    buckets = {
        name: {gate: defaultdict(lambda: {
            "clips": 0, "clips_reaching_four_frames": 0,
            "sampled_frames": 0, "accepted_frames": 0,
            "frame_reasons": Counter()
        }) for gate in GATES} for name in DETECTORS
    }
    failures = Counter()
    with zipfile.ZipFile(archive) as zipped, ExitStack() as stack:
        # Instantiate each detector only once for repeatable paired comparison.
        models = {
            name: stack.enter_context(mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=params["model_complexity"],
                min_detection_confidence=params["min_detection_confidence"]
            )) for name, params in DETECTORS.items()
        }
        members = {x.filename: x for x in zipped.infolist()}
        for row in selected:
            outcome = _inspect_video(
                members.get(row["archive_member"]), zipped, models, cv2,
                sample_fps=sample_fps, max_frames=max_frames,
                max_video_mb=max_video_mb
            )
            if "file_error" in outcome:
                failures[outcome["file_error"]] += 1
                continue
            for mode in DETECTORS:
                for gate in GATES:
                    b = buckets[mode][gate][row["label"]]
                    accepted = outcome["modes"][mode][gate]["accepted"]
                    b["clips"] += 1
                    b["clips_reaching_four_frames"] += int(accepted >= MIN_GOOD_FRAMES)
                    b["sampled_frames"] += outcome["sampled"]
                    b["accepted_frames"] += accepted
                    b["frame_reasons"].update(outcome["modes"][mode][gate])

    details = {}
    for mode in DETECTORS:
        details[mode] = {}
        for gate in GATES:
            by_label = {}
            total = Counter()
            for label in ("good", "bad"):
                b = buckets[mode][gate].get(label)
                if b is None:
                    continue
                record = {
                    "clips": b["clips"],
                    "clips_reaching_four_frames": b["clips_reaching_four_frames"],
                    "sampled_frames": b["sampled_frames"],
                    "accepted_frames": b["accepted_frames"],
                    "frame_reasons": dict(b["frame_reasons"])
                }
                by_label[label] = record
                for key in ("clips", "clips_reaching_four_frames",
                            "sampled_frames", "accepted_frames"):
                    total[key] += b[key]
            details[mode][gate] = {"total": dict(total), "by_label": by_label}

    result = {
        "study": "M22 paired front-view Push-up detection vs visibility ablation",
        "source": "Mendeley Gym7 v3",
        "split": "original M20 TRAIN subjects only, seed 42",
        "validation_or_test_decoded": False,
        "evaluation_of_form_quality": False,
        "selected_train_rejected_clips": len(selected),
        "video_file_failures": dict(failures),
        "settings": {
            "sample_fps": sample_fps,
            "max_frames_per_clip": max_frames,
            "visibility_threshold": MIN_VISIBILITY,
            "minimum_accepted_frames_per_clip": MIN_GOOD_FRAMES,
            "detectors": DETECTORS,
            "gates": {
                "full_body": "one complete shoulder-elbow-wrist-hip-knee-ankle side",
                "upper_body": "one shoulder-elbow-wrist-hip side, torso-scale check"
            }
        },
        "comparisons": details,
        "caution": (
            "The upper-body gate is a hypothesis tested on TRAIN only. "
            "Higher coverage does not establish accurate good/bad classification. "
            "Any future accepted features require a NEW schema/model, "
            "fixed participant splits, validation and independent TEST."
        )
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path(
        "data/external/mendeley_gym7/Dataset Exercise-wise.zip"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/external/mendeley_gym7/m20_manifest.csv"))
    parser.add_argument("--features", type=Path, default=Path(
        "data/features/m20_gym7/m20_clip_features.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/m22_front_gate_detector_ablation.json"))
    parser.add_argument("--limit", type=int, default=0,
                        help="TRAIN-rejected clips only; 0 = all eligible")
    args = parser.parse_args(argv)
    try:
        result = audit(args.zip, args.manifest, args.features,
                       args.output, limit=args.limit)
    except (OSError, ValueError, RuntimeError, ImportError, zipfile.BadZipFile) as e:
        print(f"M22 audit failed: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
