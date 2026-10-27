"""M23 TRAIN-only matched Push-up upper-body feature-schema experiment.

TRAIN is reconstructed exclusively from the ORIGINAL M20 accepted CSV,
NOT recalculated after recovering previously dropped clips. Validation
and TEST videos are never decoded. This is exploratory data extraction,
NOT a validated model nor public API endpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
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
from scripts.m22_train_only_pose_ablation import upper_body_gate

SCHEMA = "gym7-pushup-upper32-clip64-v1"
KEY_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24)
SIDES = ((11, 13, 15, 23), (12, 14, 16, 24))
THRESHOLD = .35
N_FEATURES = 64
FEATURE_COLUMNS = tuple(f"upper_clip_{i:02d}" for i in range(N_FEATURES))
META = ("video_id", "subject_id", "exercise", "label", "view")
COLS = (*META, *FEATURE_COLUMNS)
GATES = ("full_body", "upper_body")


def upper32_frame(landmarks, *, min_visibility=THRESHOLD):
    """One 32-feature frame vector anchored at the best-visible upper side.

    8 joints * (3 spatial coords + 1 visibility) = 32.
    Invisible joint XYZ is zero-masked; visibility is retained to prevent
    spurious coordinates from hidden joints contaminating the model.
    """
    if upper_body_gate(landmarks, min_visibility=min_visibility) != "accepted":
        return None
    points = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float64)
    visibility = np.asarray([p.visibility for p in landmarks], dtype=np.float64)
    chosen = max(SIDES, key=lambda side: min(visibility[list(side)]))
    shoulder, hip = chosen[0], chosen[-1]
    scale = float(np.linalg.norm(points[shoulder, :2] - points[hip, :2]))
    if scale < .03 or not math.isfinite(scale):
        return None
    coords = (points[list(KEY_JOINTS)] - points[hip]) / scale
    visible = visibility[list(KEY_JOINTS)]
    coords[visible < min_visibility] = 0.0
    result = np.column_stack([coords, visible]).reshape(-1)
    if result.shape != (32,) or not np.isfinite(result).all():
        return None
    return result


def aggregate_frame_vectors(vectors):
    if len(vectors) < 4:
        return None
    frames = np.asarray(vectors, dtype=np.float64)
    out = np.concatenate([np.median(frames, axis=0), np.std(frames, axis=0)])
    if out.shape != (N_FEATURES,) or not np.isfinite(out).all():
        return None
    return out


def frozen_train_manifest(source_rows, original_features, *, seed=42):
    """Select only TRAIN participant Push-ups from original M20 source."""
    if seed != 42:
        raise ValueError("Original M20 split must remain seed 42")
    parts = split_by_subject(original_features, seed=seed)
    train_ids = set(parts["train"].subject_id)
    holdout = set(parts["val"].subject_id) | set(parts["test"].subject_id)
    if train_ids & holdout:
        raise AssertionError("Participant split overlap")
    manifest_by_id = {r["video_id"]: r for r in source_rows}
    if len(manifest_by_id) != len(source_rows):
        raise ValueError("Duplicate manifest video IDs")
    for r in original_features.itertuples(index=False):
        source = manifest_by_id.get(r.video_id)
        if source is None or source["subject_id"] != r.subject_id:
            raise ValueError("Original feature metadata differs from source manifest")
    eligible = sorted(
        (r for r in source_rows if r["exercise"] == "pushup"
         and r["subject_id"] in train_ids),
        key=lambda r: r["video_id"]
    )
    if not eligible:
        raise ValueError("No original TRAIN Push-up clips available")
    return eligible, train_ids


def _extract_video(zipped, member, pose, cv2, *, sample_fps=2.,
                   max_frames=240, max_video_mb=500):
    if (member is None or member.is_dir() or
        member.file_size > max_video_mb * 1024 ** 2 or
        PurePosixPath(member.filename).suffix.lower() not in VIDEO_TYPES):
        return {"error": "missing_or_oversize_video"}
    frames = {gate: [] for gate in GATES}
    with tempfile.TemporaryDirectory(prefix="m23_train_pushup_") as folder:
        video = Path(folder) / ("clip" + PurePosixPath(member.filename).suffix.lower())
        with zipped.open(member) as source, video.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        capture = cv2.VideoCapture(str(video))
        sampled = 0
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                return {"error": "unreadable_video"}
            stride = max(1, round(fps / sample_fps))
            index = 0
            while sampled < max_frames:
                ok, image = capture.read()
                if not ok:
                    break
                index += 1
                if (index - 1) % stride:
                    continue
                sampled += 1
                result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
                # Exactly one MediaPipe inference per sampled frame.
                full_reason, _ = diagnose_pushup_pose(landmarks)
                upper_reason = upper_body_gate(landmarks)
                if upper_reason != "accepted":
                    continue
                vector = upper32_frame(landmarks)
                if vector is None:
                    continue
                frames["upper_body"].append(vector)
                if full_reason == "accepted":
                    frames["full_body"].append(vector)
        finally:
            capture.release()
    return {"sampled": sampled,
            "clip_vectors": {
                gate: aggregate_frame_vectors(rows) for gate, rows in frames.items()
            },
            "accepted_frame_counts": {gate: len(rows) for gate, rows in frames.items()}}


def extract_train(archive: Path, manifest: Path, original_csv: Path, output_dir: Path,
                  *, max_clips=0, sample_fps=2., max_frames=240,
                  max_video_mb=500):
    if (output_dir.exists() or max_clips < 0 or
        not 0 < sample_fps <= 10 or not 4 <= max_frames <= 2000 or
        max_video_mb < 5):
        raise ValueError("Output already exists or invalid extraction limits")

    rows = read_manifest(manifest)
    original = load_clip_table(original_csv)
    chosen, subjects = frozen_train_manifest(rows, original)
    if max_clips:
        chosen = chosen[:max_clips]

    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError("Activate the isolated .venv-m20 environment") from error

    by_gate = {gate: [] for gate in GATES}
    by_label_view = {gate: Counter() for gate in GATES}
    failed = Counter()
    with zipfile.ZipFile(archive) as zipped:
        members = {item.filename: item for item in zipped.infolist()}
        # Reuse the original M20 detector model (complexity 1, conf 0.5).
        with mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=1,
            min_detection_confidence=.5
        ) as pose:
            for row in chosen:
                data = _extract_video(zipped, members.get(row["archive_member"]),
                                      pose, cv2, sample_fps=sample_fps,
                                      max_frames=max_frames, max_video_mb=max_video_mb)
                if "error" in data:
                    failed[data["error"]] += 1
                    continue
                for gate in GATES:
                    vector = data["clip_vectors"][gate]
                    if vector is None:
                        continue
                    by_gate[gate].append({
                        **{key: row[key] for key in META},
                        **dict(zip(FEATURE_COLUMNS, map(float, vector)))
                    })
                    by_label_view[gate][(row["label"], row["view"])] += 1

    if not any(by_gate.values()):
        raise ValueError("No valid TRAIN-only Push-up clips in new schema")
    # Avoid writes until all selected video processing succeeds.
    output_dir.mkdir(parents=True, exist_ok=False)
    for gate, records in by_gate.items():
        with (output_dir / f"train_{gate}.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLS)
            writer.writeheader()
            writer.writerows(records)
    report = {
        "study": "M23 new masked 64-feature Push-up video-quality schema",
        "schema": SCHEMA,
        "dataset": "Mendeley Gym7 v3",
        "split": "frozen M20 seed=42 TRAIN subjects only",
        "validation_or_test_video_decoded": False,
        "original_m20_subjects": len(subjects),
        "selected_train_pushup_clips": len(chosen),
        "sample_fps": sample_fps,
        "max_frames": max_frames,
        "gate_counts": {gate: {
            "accepted_clips": len(by_gate[gate]),
            "by_label_view": {
                f"{label}/{view}": int(count)
                for (label, view), count in sorted(by_label_view[gate].items())
            },
            "file": f"train_{gate}.csv"
        } for gate in GATES},
        "file_failures": dict(failed),
        "caution": (
            "TRAIN only. These are video-level publisher good/bad features, "
            "not anatomical form errors. No classifier accuracy measured. "
            "Do not retrain legacy 264-feature RF with 64-feature inputs. "
            "Keep original M20 holdout subject membership unchanged."
        )
    }
    (output_dir / "train_extract_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", type=Path, default=Path(
        "data/external/mendeley_gym7/Dataset Exercise-wise.zip"))
    p.add_argument("--manifest", type=Path, default=Path(
        "data/external/mendeley_gym7/m20_manifest.csv"))
    p.add_argument("--original-features", type=Path, default=Path(
        "data/features/m20_gym7/m20_clip_features.csv"))
    p.add_argument("--output-dir", type=Path, default=Path(
        "data/features/m23_pushup_train_upper_v1"))
    p.add_argument("--limit", type=int, default=0)
    options = p.parse_args(argv)
    try:
        result = extract_train(
            options.zip, options.manifest, options.original_features,
            options.output_dir, max_clips=options.limit
        )
    except (ValueError, OSError, RuntimeError, ImportError,
            zipfile.BadZipFile) as error:
        print(f"M23 extraction failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
