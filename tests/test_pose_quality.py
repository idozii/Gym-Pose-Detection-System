from types import SimpleNamespace

import pytest

from gym_pose.quality import assess_pose_quality


def pose_landmarks():
    coords = []
    for idx in range(33):
        coords.append(SimpleNamespace(x=0.01 * idx, y=0.02 * idx,
                                      z=0.0, visibility=1.0))
    return coords


def test_no_pose_reason():
    assert assess_pose_quality(None)["reason"] == "no_pose_detected"


def test_valid_pose_matches_core_gate():
    result = assess_pose_quality(pose_landmarks())
    assert result["reason"] == "accepted"
    assert result["low_visibility_joints"] == []
    assert result["minimum_critical_joint_visibility"] == 1.0


def test_critical_knee_visibility_reason_and_joint():
    joints = pose_landmarks()
    joints[25].visibility = 0.3
    result = assess_pose_quality(joints)
    assert result["reason"] == "critical_joint_low_visibility"
    assert "left_knee" in result["low_visibility_joints"]
    assert assess_pose_quality(joints, min_visibility=.25)["reason"] == "accepted"


def test_invalid_landmark_count_and_nonfinite():
    assert assess_pose_quality(pose_landmarks()[:32])["reason"] == "invalid_landmark_structure"
    joints = pose_landmarks()
    joints[24].x = float("nan")
    assert assess_pose_quality(joints)["reason"] == "nonfinite_coordinates_or_visibility"


def test_zero_shoulder_distance():
    joints = pose_landmarks()
    joints[12].x, joints[12].y = joints[11].x, joints[11].y
    assert assess_pose_quality(joints)["reason"] == "degenerate_shoulder_distance"


def test_invalid_min_visibility():
    with pytest.raises(ValueError):
        assess_pose_quality(pose_landmarks(), min_visibility=1.2)
