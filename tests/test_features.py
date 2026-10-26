from types import SimpleNamespace

import numpy as np
import pytest

from gym_pose.features import FEATURE_NAMES, SCHEMA_VERSION, calculate_angle, pose_to_features


def sample_landmarks():
    coords = []
    for i in range(33):
        x, y = (i % 8) / 10, (i // 8) / 10
        coords.append(SimpleNamespace(x=x, y=y, z=0.0, visibility=1.0))
    return coords


def test_angle():
    assert calculate_angle((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0)


def test_schema_and_names():
    features = pose_to_features(sample_landmarks())
    assert SCHEMA_VERSION == "squat-103-v1"
    assert len(FEATURE_NAMES) == len(features) == 103
    assert list(features) == list(FEATURE_NAMES)
    assert all(np.isfinite(list(features.values())))


def test_low_visibility_abstains():
    landmarks = sample_landmarks()
    landmarks[25].visibility = .1
    assert pose_to_features(landmarks) is None


def test_bad_shape_and_nonfinite_abstain():
    assert pose_to_features(sample_landmarks()[:32]) is None
    landmarks = sample_landmarks()
    landmarks[23].x = float("nan")
    assert pose_to_features(landmarks) is None


def test_degenerate_shoulders_abstains():
    landmarks = sample_landmarks()
    landmarks[12].x, landmarks[12].y = landmarks[11].x, landmarks[11].y
    assert pose_to_features(landmarks) is None
