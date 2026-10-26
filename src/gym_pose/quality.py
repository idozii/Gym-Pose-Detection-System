"""Detailed, read-only diagnostics for the existing squat landmark gate.

The production feature extractor is authoritative. This module does not change
its thresholds, its 103-feature schema, or the trained model. All explanations
are observational and are not evidence of biomechanical correctness.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .features import IMPORTANT_INDICES, LANDMARK_COUNT, pose_to_features

JOINT_NAMES = {
    11: "left_shoulder", 12: "right_shoulder",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}
REASONS = (
    "accepted", "no_pose_detected", "invalid_landmark_structure",
    "nonfinite_coordinates_or_visibility", "critical_joint_low_visibility",
    "degenerate_shoulder_distance", "other_feature_rejection",
)


def assess_pose_quality(landmarks: Any, *, min_visibility: float = 0.5) -> dict:
    """Explain failures in exactly the order of the existing feature gate."""
    if not 0.0 <= min_visibility <= 1.0:
        raise ValueError("min_visibility must be between 0 and 1")
    result = {"reason": "accepted", "low_visibility_joints": [],
              "minimum_critical_joint_visibility": None,
              "shoulder_distance": None}
    if landmarks is None:
        result["reason"] = "no_pose_detected"
        return result
    try:
        if len(landmarks) != LANDMARK_COUNT:
            result["reason"] = "invalid_landmark_structure"
            return result
        coords = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
        vis = np.asarray([float(getattr(p, "visibility", 0.0))
                          for p in landmarks], dtype=float)
    except (TypeError, ValueError, AttributeError):
        result["reason"] = "invalid_landmark_structure"
        return result
    if coords.shape != (LANDMARK_COUNT, 3) or not np.isfinite(coords).all() or not np.isfinite(vis).all():
        result["reason"] = "nonfinite_coordinates_or_visibility"
        return result
    key_vis = vis[list(IMPORTANT_INDICES)]
    result["minimum_critical_joint_visibility"] = float(key_vis.min())
    result["low_visibility_joints"] = [
        JOINT_NAMES[i] for i in IMPORTANT_INDICES if vis[i] < min_visibility
    ]
    if result["low_visibility_joints"]:
        result["reason"] = "critical_joint_low_visibility"
        return result
    distance = float(np.linalg.norm(coords[11] - coords[12]))
    result["shoulder_distance"] = distance
    if distance <= 1e-6:
        result["reason"] = "degenerate_shoulder_distance"
        return result
    if pose_to_features(landmarks, min_visibility=min_visibility) is None:
        result["reason"] = "other_feature_rejection"
    return result
