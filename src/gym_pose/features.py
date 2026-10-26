"""Backward-compatible 103-dimensional MediaPipe Pose feature schema.

The four engineered features intentionally retain the original notebook's
definitions. In particular torso_incline_deg is camera-relative, NOT a
clinically meaningful angle or a camera-invariant trunk inclination.
"""
from __future__ import annotations

from typing import Any

import numpy as np

SCHEMA_VERSION = "squat-103-v1"
LANDMARK_COUNT = 33
FEATURE_NAMES = tuple(
    f"lm_{i}_{axis}" for i in range(LANDMARK_COUNT) for axis in ("x", "y", "z")
) + ("knee_angle_mean", "hip_angle_mean", "torso_incline_deg", "hip_to_shoulder_ratio")
IMPORTANT_INDICES = (11, 12, 23, 24, 25, 26, 27, 28)


def calculate_angle(a: Any, b: Any, c: Any) -> float:
    """Return angle ABC in degrees, matching the original notebook formula."""
    aa, bb, cc = (np.asarray(point, dtype=float)[:2] for point in (a, b, c))
    radians = np.arctan2(cc[1] - bb[1], cc[0] - bb[0]) - np.arctan2(
        aa[1] - bb[1], aa[0] - bb[0]
    )
    angle = abs(float(np.degrees(radians)))
    return 360.0 - angle if angle > 180.0 else angle


def pose_to_features(landmarks: Any, *, min_visibility: float = 0.5) -> dict[str, float] | None:
    """Return named features; None denotes an unreliable or unavailable pose.

    MediaPipe landmarks are expected in the original fixed 33-landmark order.
    A low-visibility critical joint causes abstention, not a guessed label.
    """
    if not 0.0 <= min_visibility <= 1.0:
        raise ValueError("min_visibility must be in [0, 1]")
    try:
        if len(landmarks) != LANDMARK_COUNT:
            return None
        coords = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
        vis = np.asarray([float(getattr(p, "visibility", 0.0)) for p in landmarks], dtype=float)
    except (TypeError, ValueError, AttributeError):
        return None
    if coords.shape != (33, 3) or not np.isfinite(coords).all() or not np.isfinite(vis).all():
        return None
    if (vis[list(IMPORTANT_INDICES)] < min_visibility).any():
        return None

    hip_center = (coords[23] + coords[24]) / 2.0
    distance = float(np.linalg.norm(coords[11] - coords[12]))
    if distance <= 1e-6:
        return None
    shoulder_dist = distance + 1e-6  # preserve the notebook's normalization
    norm_coords = (coords - hip_center) / shoulder_dist

    left_knee = calculate_angle(coords[23], coords[25], coords[27])
    right_knee = calculate_angle(coords[24], coords[26], coords[28])
    left_hip = calculate_angle(coords[11], coords[23], coords[25])
    right_hip = calculate_angle(coords[12], coords[24], coords[26])
    torso_vec = coords[11] - coords[23]
    torso_incline = float(np.degrees(np.arctan2(torso_vec[1], torso_vec[0])))
    engineered = np.array([
        (left_knee + right_knee) / 2.0,
        (left_hip + right_hip) / 2.0,
        torso_incline,
        np.linalg.norm(coords[23] - coords[24]) / shoulder_dist,
    ], dtype=np.float32)
    vector = np.concatenate((norm_coords.flatten(), engineered))
    if vector.shape != (103,) or not np.isfinite(vector).all():
        return None
    return dict(zip(FEATURE_NAMES, map(float, vector)))
