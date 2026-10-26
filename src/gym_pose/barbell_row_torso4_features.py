"""Camera-relative four-joint Barbell Row static features.

Separate versioned schema from the legacy eight-joint/32-feature baseline.
Uses shoulders and hips ONLY, including for every engineered feature.
In particular, MediaPipe's four body landmarks CANNOT measure spinal
curvature and do not provide direct clinical lumbar safety assessment.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from .features import LANDMARK_COUNT

SCHEMA_VERSION = "barbell-row-torso4-image-20-v1"
REQUIRED_JOINTS = (11, 12, 23, 24)
EXTRA_NAMES = (
    "left_trunk_direction_deg",
    "right_trunk_direction_deg",
    "shoulder_y_asymmetry_norm",
    "hip_y_asymmetry_norm",
    "trunk_length_over_shoulder_width",
    "hip_width_over_shoulder_width",
    "shoulder_minus_hip_center_x_norm",
    "shoulder_minus_hip_center_y_norm",
)
FEATURE_NAMES = tuple(
    f"lm_{index}_{axis}"
    for index in REQUIRED_JOINTS
    for axis in ("x", "y", "z")
) + EXTRA_NAMES


def pose_to_torso4_features(
    landmarks: Any, *, min_visibility: float = .5
) -> dict[str, float] | None:
    """20 finite features or None; never access elbows, knees, or ankles."""
    if not np.isfinite(min_visibility) or not 0 <= min_visibility <= 1:
        raise ValueError("min_visibility must be finite in [0,1]")
    try:
        if len(landmarks) != LANDMARK_COUNT:
            return None
        xyz = np.array([
            (landmarks[i].x, landmarks[i].y, landmarks[i].z)
            for i in REQUIRED_JOINTS
        ], dtype=float)
        visibility = np.asarray([
            float(landmarks[i].visibility) for i in REQUIRED_JOINTS
        ], dtype=float)
    except (IndexError, ValueError, TypeError, AttributeError):
        return None
    if (xyz.shape != (4, 3) or not np.isfinite(xyz).all()
        or not np.isfinite(visibility).all()
        or (visibility < min_visibility).any()):
        return None
    p = dict(zip(REQUIRED_JOINTS, xyz))
    shoulder_width = float(np.linalg.norm(p[11] - p[12]))
    if shoulder_width <= 1e-6:
        return None
    hip_center = (p[23] + p[24]) / 2.
    shoulder_center = (p[11] + p[12]) / 2.
    scale = shoulder_width + 1e-6
    norm_xyz = ((xyz - hip_center) / scale).reshape(-1)
    left = p[11] - p[23]
    right = p[12] - p[24]
    extras = (
        float(np.degrees(np.arctan2(left[1], left[0]))),
        float(np.degrees(np.arctan2(right[1], right[0]))),
        float((p[11][1] - p[12][1]) / scale),
        float((p[23][1] - p[24][1]) / scale),
        float(np.linalg.norm(shoulder_center - hip_center) / scale),
        float(np.linalg.norm(p[23] - p[24]) / scale),
        float((shoulder_center[0] - hip_center[0]) / scale),
        float((shoulder_center[1] - hip_center[1]) / scale),
    )
    values = np.r_[norm_xyz, np.asarray(extras, dtype=float)]
    if values.shape != (20,) or not np.isfinite(values).all():
        return None
    return dict(zip(FEATURE_NAMES, map(float, values)))
