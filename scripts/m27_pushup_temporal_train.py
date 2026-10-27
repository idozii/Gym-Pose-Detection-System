"""M27: TRAIN-only temporal Push-up hypothesis, never original VAL/TEST.

Adds time-ordered elbow flexion and wrist/shoulder trajectory summaries to
the SAME admitted M23 upper-body clips. Paired GroupKFold compares a fixed
200-tree RF trained on the original 64 features to the same RF trained on
64+16 features. Good/bad labels come from the publisher's VIDEO labels.

Research-only. No new website classifier or source error diagnosis.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from gym_pose.gym7_research import read_manifest, VIDEO_TYPES
from gym_pose.gym7_training import load_clip_table
from gym_pose.gym7_m24_cv import _metrics, _read_table
from scripts.m23_train_only_pushup_features import (
    KEY_JOINTS, SIDES, THRESHOLD, FEATURE_COLUMNS, META,
    frozen_train_manifest, upper32_frame,
)

SCHEMA = "gym7-pushup-temporal16-v1"
MOTION = (
    "elbow_angle_p10_deg", "elbow_angle_median_deg",
    "elbow_angle_p90_deg", "elbow_angle_std_deg",
    "elbow_angle_range_deg", "elbow_speed_mean_deg_s",
    "elbow_speed_p90_deg_s", "elbow_direction_switch_rate",
    "elbow_first_last_delta_deg", "wrist_shoulder_distance_median",
    "wrist_shoulder_distance_std", "wrist_shoulder_distance_range",
    "valid_span_seconds", "accepted_frame_fraction",
    "valid_frame_count", "selected_arm_side_switch_rate",
)
MOTION_COLS = tuple("motion_" + name for name in MOTION)
ALL_COLS = (*META, *MOTION_COLS)


def motion_frame(landmarks):
    """Return elbow angle, normalized wrist distance, visible side ID.

    This is only defined on an already accepted M23 upper-body frame.
    No imputation of completely absent pose landmarks is performed.
    """
    if upper32_frame(landmarks) is None:
        return None
    points = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=float)
    vis = np.asarray([p.visibility for p in landmarks], dtype=float)
    side_id = int(np.argmax([min(vis[list(side)]) for side in SIDES]))
    shoulder, elbow, wrist, _hip = SIDES[side_id]
    a = points[shoulder, :2] - points[elbow, :2]
    b = points[wrist, :2] - points[elbow, :2]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return None
    cos_value = np.clip(float(np.dot(a, b) / denom), -1., 1.)
    angle = float(np.degrees(np.arccos(cos_value)))
    sh_center = (points[11] + points[12]) / 2
    hip_center = (points[23] + points[24]) / 2
    torso_scale = float(np.linalg.norm((sh_center - hip_center)[:2]))
    if torso_scale < .03 or not math.isfinite(torso_scale):
        return None
    wrist_distance = float(
        np.linalg.norm(points[wrist, :2] - points[shoulder, :2]) / torso_scale
    )
    if not np.isfinite([angle, wrist_distance]).all():
        return None
    return angle, wrist_distance, side_id


def motion_summary(sequence, sampled_frames):
    """16 time-ordered descriptors from (t_seconds, angle, distance, side).

    Time-normalized speeds respect missing-frame gaps; direction switching
    requires at least 5 deg/s to suppress small numerical fluctuations.
    """
    if sampled_frames < 4 or len(sequence) < 4:
        return None
    values = np.asarray(sequence, dtype=float)
    if values.shape != (len(sequence), 4) or not np.isfinite(values).all():
        return None
    t, angle, wrist, side = values.T
    if np.any(np.diff(t) <= 0) or t[-1] <= t[0]:
        return None
    durations = np.diff(t)
    slopes = np.diff(angle) / durations
    speed = np.abs(slopes)
    substantial = np.sign(slopes[np.abs(slopes) >= 5.])
    turns = (np.count_nonzero(np.diff(substantial) != 0) /
             max(1, len(substantial) - 1))
    arm_switch = float(np.count_nonzero(np.diff(side) != 0)) / max(1, len(side)-1)
    arr = np.asarray((
        np.percentile(angle, 10), np.median(angle), np.percentile(angle, 90),
        np.std(angle), np.ptp(angle), np.mean(speed), np.percentile(speed, 90),
        turns, angle[-1] - angle[0],
        np.median(wrist), np.std(wrist), np.ptp(wrist),
        t[-1]-t[0], len(sequence)/sampled_frames,
        len(sequence), arm_switch,
    ), dtype=float)
    if arr.shape != (16,) or not np.isfinite(arr).all():
        return None
    return arr


def _read_video(zipped, member, detector, cv2, *, sample_fps=2.,
                max_frames=240, max_video_mb=500):
    if (member is None or member.is_dir() or
        member.file_size > max_video_mb*1024*1024 or
        PurePosixPath(member.filename).suffix.lower() not in VIDEO_TYPES):
        return None, "missing_or_oversize_video"
    timeline = []
    sampled = 0
    with tempfile.TemporaryDirectory(prefix="m27_train_only_") as directory:
        file = Path(directory) / ("clip" + PurePosixPath(member.filename).suffix.lower())
        with zipped.open(member) as src, file.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024*1024)
        capture = cv2.VideoCapture(str(file))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                return None, "invalid_video_fps_or_decode"
            stride = max(1, round(fps/sample_fps))
            frame_index = 0
            while sampled < max_frames:
                has_frame, frame = capture.read()
                if not has_frame:
                    break
                frame_index += 1
                if (frame_index - 1) % stride:
                    continue
                sampled += 1
                result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = (
                    result.pose_landmarks.landmark
                    if result.pose_landmarks else None
                )
                row = motion_frame(landmarks)
                if row is not None:
                    # Use original frame timestamp, not accepted-frame index.
                    timeline.append(((frame_index-1)/fps, *row))
        finally:
            capture.release()
    return motion_summary(timeline, sampled), None


def extract(archive: Path, manifest_path: Path, original_path: Path,
            existing_upper_path: Path, destination: Path, *,
            sample_fps=2., max_frames=240, max_video_mb=500):
    if destination.exists():
        raise FileExistsError("Existing M27 feature output cannot be overwritten")
    if (not 0 < sample_fps <= 10 or not 4 <= max_frames <= 2000
        or max_video_mb < 5):
        raise ValueError("Invalid source video sampling controls")
    manifest = read_manifest(manifest_path)
    original = load_clip_table(original_path)
    train_rows, train_people = frozen_train_manifest(manifest, original)
    permitted = {r["video_id"]: r for r in train_rows}
    upper = _read_table(existing_upper_path)
    if not set(upper.index).issubset(permitted):
        raise ValueError("M23 upper table contains non-TRAIN or unknown videos")
    chosen = []
    for vid in sorted(upper.index):
        record = permitted[vid]
        if (record["subject_id"] not in train_people or
            any(upper.at[vid, key] != record[key] for key in META)):
            raise ValueError("M23 upper metadata differs from frozen TRAIN manifest")
        chosen.append(record)

    try:
        import cv2
        import mediapipe as mp
    except ImportError as err:
        raise RuntimeError("Use the original .venv-m20 MediaPipe environment") from err

    output_rows = []
    fail = Counter()
    with zipfile.ZipFile(archive) as zipped:
        members = {z.filename: z for z in zipped.infolist()}
        with mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=1,
            min_detection_confidence=.5
        ) as detector:
            for record in chosen:
                summary, error = _read_video(
                    zipped, members.get(record["archive_member"]),
                    detector, cv2, sample_fps=sample_fps,
                    max_frames=max_frames, max_video_mb=max_video_mb
                )
                if error or summary is None:
                    fail[error or "insufficient_motion_samples"] += 1
                    continue
                output_rows.append({
                    **{k:record[k] for k in META},
                    **dict(zip(MOTION_COLS, map(float, summary)))
                })

    if len(output_rows) != len(chosen):
        raise ValueError(
            f"Temporal extraction produced {len(output_rows)}/{len(chosen)} "
            f"original M23 upper clips. Reasons: {dict(fail)}. "
            "No partial research file was written; investigate detector or "
            "angle visibility before changing the paired evaluation cohort."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS)
        writer.writeheader()
        writer.writerows(output_rows)
    return {
        "schema": SCHEMA, "original_seed": 42,
        "original_frozen_TRAIN_only": True,
        "val_or_test_source_video_decoded": False,
        "matched_original_M23_upper_videos": len(chosen),
        "temporal_features": len(MOTION_COLS),
        "output": str(destination),
        "label_semantics": "publisher video-level good/bad only",
    }


def _read_temporal(path):
    df = pd.read_csv(path, keep_default_na=False)
    if tuple(df.columns) != ALL_COLS or df.empty or df.video_id.duplicated().any():
        raise ValueError("M27 temporal feature schema mismatch or duplicated video")
    vals = df.loc[:, MOTION_COLS].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("M27 motion values must be finite numeric values")
    return df.set_index("video_id", drop=False)


def compare(temporal_path, upper_path, manifest_path, original_path, *,
            folds=5, trees=200):
    if not 2 <= folds <= 14 or trees != 200:
        raise ValueError("Keep M27 5-fold, 200-tree preregistered defaults")
    manifest = read_manifest(manifest_path)
    original = load_clip_table(original_path)
    train_rows, train_people = frozen_train_manifest(manifest, original)
    permitted = {r["video_id"]: r for r in train_rows}

    baseline = _read_table(upper_path)
    temporal = _read_temporal(temporal_path)
    if set(baseline.index) != set(temporal.index):
        raise ValueError("M27 temporal and M23 baseline must have identical video IDs")
    for frame in (baseline, temporal):
        if not set(frame.index).issubset(permitted):
            raise ValueError("Non-TRAIN person/video found in M27 comparison")
        for vid in frame.index:
            source = permitted[vid]
            if (source["subject_id"] not in train_people or
                any(frame.at[vid, key] != source[key] for key in META)):
                raise ValueError("Source TRAIN metadata mismatch")
    ids = sorted(baseline.index)
    base = baseline.loc[ids]
    merged = base.loc[:, (*META, *FEATURE_COLUMNS)].join(
        temporal.loc[ids, MOTION_COLS], how="inner"
    )
    if len(merged) != len(ids) or merged.subject_id.nunique() < folds:
        raise ValueError("CV identities or subject groups missing")
    expected = np.asarray([int(base.at[vid,"label"]=="bad") for vid in ids])
    names = {"baseline_64": FEATURE_COLUMNS,
             "temporal_80": (*FEATURE_COLUMNS, *MOTION_COLS)}
    results = {name:{} for name in names}
    folds_actual=0
    for train_idx, val_idx in GroupKFold(n_splits=folds).split(
        merged, groups=merged.subject_id
    ):
        folds_actual += 1
        people_train = set(merged.iloc[train_idx].subject_id)
        people_hold = set(merged.iloc[val_idx].subject_id)
        if people_train & people_hold:
            raise AssertionError("M27 CV person leakage")
        yt = expected[train_idx]
        if set(yt) != {0,1}:
            raise ValueError("TRAIN fold missing original publisher quality class")
        for name, cols in names.items():
            model = RandomForestClassifier(
                n_estimators=trees, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=1
            )
            model.fit(merged.iloc[train_idx].loc[:,cols].to_numpy(dtype=float), yt)
            p = model.predict_proba(
                merged.iloc[val_idx].loc[:,cols].to_numpy(dtype=float)
            )[:, list(model.classes_).index(1)]
            for k, prediction in zip(val_idx, p):
                if k in results[name]:
                    raise AssertionError("Repeated CV prediction")
                results[name][int(k)] = float(prediction)
    if any(set(res) != set(range(len(ids))) for res in results.values()):
        raise AssertionError("OOF predictions incomplete")
    scored = {}
    for name, predicted in results.items():
        pairs = [(int(expected[i]), predicted[i]) for i in range(len(ids))]
        scored[name] = {
            "all_paired": _metrics(pairs),
            "by_view": {
                view: _metrics([
                    pairs[i] for i in range(len(ids)) if base.iloc[i].view==view
                ]) for view in ("front","side","diagonal")
            }
        }
    return {
        "study": "M27 TRAIN-only paired time-order hypothesis",
        "schema": SCHEMA,
        "m23_baseline_schema": "gym7-pushup-upper32-clip64-v1",
        "models": ("baseline_64", "temporal_80"),
        "subjects": int(merged.subject_id.nunique()),
        "paired_clips": len(ids),
        "group_kfold_folds": folds_actual,
        "fixed_rf_trees": trees,
        "metrics": scored,
        "original_M20_VAL_or_TEST_used": False,
        "warning": (
            "Post-M26 exploratory TRAIN-only temporal analysis; "
            "future comparison on M20 VAL would be adaptive, not confirmatory. "
            "Do not claim gym-form correctness or use original TEST."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    extraction = sub.add_parser("extract")
    extraction.add_argument("--zip", type=Path, default=Path(
        "data/external/mendeley_gym7/Dataset Exercise-wise.zip"))
    extraction.add_argument("--manifest", type=Path, default=Path(
        "data/external/mendeley_gym7/m20_manifest.csv"))
    extraction.add_argument("--original", type=Path, default=Path(
        "data/features/m20_gym7/m20_clip_features.csv"))
    extraction.add_argument("--upper", type=Path, default=Path(
        "data/features/m23_pushup_train_upper_v1/train_upper_body.csv"))
    extraction.add_argument("--output", type=Path, default=Path(
        "data/features/m27_pushup_train_motion_v1.csv"))
    cv = sub.add_parser("compare")
    cv.add_argument("--motion", type=Path, default=Path(
        "data/features/m27_pushup_train_motion_v1.csv"))
    cv.add_argument("--upper", type=Path, default=Path(
        "data/features/m23_pushup_train_upper_v1/train_upper_body.csv"))
    cv.add_argument("--manifest", type=Path, default=Path(
        "data/external/mendeley_gym7/m20_manifest.csv"))
    cv.add_argument("--original", type=Path, default=Path(
        "data/features/m20_gym7/m20_clip_features.csv"))
    cv.add_argument("--report", type=Path, default=Path(
        "data/processed/m27_pushup_temporal_train_cv.json"))
    opts = parser.parse_args(argv)
    try:
        if opts.action == "extract":
            summary = extract(
                opts.zip, opts.manifest, opts.original, opts.upper, opts.output
            )
        else:
            if opts.report.exists():
                raise FileExistsError("M27 CV report exists, will never overwrite")
            summary = compare(opts.motion, opts.upper, opts.manifest, opts.original)
            opts.report.parent.mkdir(parents=True, exist_ok=True)
            opts.report.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
    except (OSError, ValueError, RuntimeError, AssertionError,
            zipfile.BadZipFile) as exc:
        print(f"M27 {opts.action} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
