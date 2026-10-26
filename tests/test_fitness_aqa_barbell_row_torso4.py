"""Synthetic checks for Row torso4 source split safety & features.

No publisher ZIP, no source TEST pixel decoding, no MediaPipe GPU required.
"""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gym_pose.barbell_row_features import pose_to_barbell_row_features
from gym_pose.barbell_row_torso4_features import (
    REQUIRED_JOINTS, FEATURE_NAMES, SCHEMA_VERSION,
    pose_to_torso4_features,
)
from scripts import extract_fitness_aqa_barbell_row as extractor
from scripts import extract_fitness_aqa_barbell_row_torso4 as cli
from tests.test_fitness_aqa_barbell_row_extraction import (
    FakeCv2, pose_points, source_fixture,
)


def test_new_20_feature_schema_does_not_use_leg_or_elbow_coordinates():
    landmarks = pose_points(low=(25, 26, 27, 28), invalid_unused=True)
    for idx in (13, 14, 15, 16, 25, 26, 27, 28):
        landmarks[idx].x = float("nan")
        landmarks[idx].y = float("nan")
        landmarks[idx].z = float("nan")
    assert pose_to_barbell_row_features(landmarks) is None
    result = pose_to_torso4_features(landmarks)
    assert result is not None
    assert REQUIRED_JOINTS == (11, 12, 23, 24)
    assert SCHEMA_VERSION == "barbell-row-torso4-image-20-v1"
    assert len(result) == 20
    assert tuple(result) == FEATURE_NAMES
    assert np.isfinite(list(result.values())).all()
    # Deliberately ruin every irrelevant landmark: outputs stay the same.
    for i in range(33):
        if i not in REQUIRED_JOINTS:
            landmarks[i].visibility = 0.
    assert pose_to_torso4_features(landmarks) == result


def test_torso4_rejects_bad_required_joints_and_geometry():
    assert pose_to_torso4_features(None) is None
    assert pose_to_torso4_features(pose_points()[:30]) is None
    assert pose_to_torso4_features(pose_points(low=(11,))) is None
    assert pose_to_torso4_features(pose_points(low=(23,))) is None
    points = pose_points()
    points[12].x = points[11].x
    points[12].y = points[11].y
    points[12].z = points[11].z
    assert pose_to_torso4_features(points) is None
    with pytest.raises(ValueError, match="min_visibility"):
        pose_to_torso4_features(pose_points(), min_visibility=float("nan"))
    with pytest.raises(ValueError, match="min_visibility"):
        pose_to_torso4_features(pose_points(), min_visibility=1.2)


class LegsOccludedPose:
    def __init__(self, **kwargs):
        assert kwargs["static_image_mode"] is True
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def process(self, _):
        return SimpleNamespace(pose_landmarks=SimpleNamespace(
            landmark=pose_points(low=(25, 26, 27, 28))
        ))


MP = SimpleNamespace(
    solutions=SimpleNamespace(
        pose=SimpleNamespace(Pose=LegsOccludedPose)))


def test_new_extractor_retains_occluded_leg_images_and_preserves_legacy(tmp_path):
    root = source_fixture(tmp_path)
    output = tmp_path / "features"
    output.mkdir()
    sentinel = output / "fitness_aqa_barbell_row_torso_angle_train_pose.csv"
    sentinel.write_text("DO NOT MODIFY LEGACY 32-FEATURE CSV")
    legacy_summary = output / "fitness_aqa_barbell_row_extraction_summary.json"
    legacy_summary.write_text('{"schema_version":"barbell-row-image-32-v1"}')
    FakeCv2.seen = []
    report = extractor.extract(
        root / "images.zip", root / "annotations", output,
        feature_profile="torso4", max_images_per_task_split=0,
        cv2_api=FakeCv2, mp_api=MP,
    )
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["feature_profile"] == "torso4"
    assert report["feature_count"] == 20
    assert report["source_test_images_decoded"] == 0
    assert sentinel.read_text() == "DO NOT MODIFY LEGACY 32-FEATURE CSV"
    assert legacy_summary.read_text() == '{"schema_version":"barbell-row-image-32-v1"}'
    assert len(FakeCv2.seen) == 7
    assert not any(key.startswith("90") for key in FakeCv2.seen)
    for split, n in (("train", 4), ("val", 3)):
        assert report["splits"][split]["images_decoded"] == n
        for task in ("torso_angle", "lumbar"):
            stats = report["splits"][split]["tasks"][task]
            assert stats["selected_images"] == n
            assert stats["pose_accepted"] == n
            assert stats["positive_accepted"] + stats["negative_accepted"] == n
            csvfile = output / f"fitness_aqa_barbell_row_torso4_{task}_{split}_pose.csv"
            with csvfile.open(newline="") as f:
                data = list(csv.DictReader(f))
            assert len(data) == n
            assert tuple(data[0]) == (*extractor.META, *FEATURE_NAMES)
            assert {row["split"] for row in data} == {split}
            assert {row["target"] for row in data} == {task}
    assert (output / "fitness_aqa_barbell_row_torso4_extraction_summary.json").is_file()
    assert not list(output.glob("*test_pose.csv"))


def test_original_extractor_defaults_remain_eight_joint_32_features(tmp_path):
    root = source_fixture(tmp_path)
    report = extractor.extract(
        root / "images.zip", root / "annotations", tmp_path / "legacy",
        max_images_per_task_split=0, cv2_api=FakeCv2, mp_api=MP)
    assert report["schema_version"] == "barbell-row-image-32-v1"
    assert report["feature_profile"] == "full8"
    assert report["feature_count"] == 32
    for split in ("train", "val"):
        for task in ("torso_angle", "lumbar"):
            assert report["splits"][split]["tasks"][task]["pose_accepted"] == 0


def test_explicit_test_selection_and_unknown_feature_profile_blocked(tmp_path):
    with pytest.raises(ValueError, match="TEST locked"):
        extractor.extract(
            tmp_path, tmp_path, tmp_path,
            selected_splits=("train", "test"), feature_profile="torso4")
    with pytest.raises(ValueError, match="Unknown Barbell Row"):
        extractor.extract(
            tmp_path, tmp_path, tmp_path, feature_profile="unknown")
