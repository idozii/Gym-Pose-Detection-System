"""Barbell Row *image* pose features; not video/temporal or medical geometry.

Both independently annotated targets use the same versioned 32-dimensional
static pose vector. MediaPipe Pose provides no directly annotated lumbar-spine
curvature landmark, so lumbar-error classification is only a surrogate
empirical association, NOT a verified measurement of spinal alignment.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from .features import calculate_angle, LANDMARK_COUNT

SCHEMA_VERSION = "barbell-row-image-32-v1"
CRITICAL_JOINTS = (11, 12, 23, 24, 25, 26, 27, 28)
EXTRA_NAMES = (
    "left_hip_angle_deg", "right_hip_angle_deg",
    "left_knee_angle_deg", "right_knee_angle_deg",
    "left_trunk_direction_deg", "right_trunk_direction_deg",
    "hip_width_over_shoulder_distance",
    "shoulder_centre_minus_hip_centre_x_norm",
)
FEATURE_NAMES = tuple(
    f"lm_{i}_{axis}" for i in CRITICAL_JOINTS
    for axis in ("x", "y", "z")
) + EXTRA_NAMES
TARGETS = ("torso_angle", "lumbar")


def pose_to_barbell_row_features(
    landmarks: Any, *, min_visibility: float = .5
) -> dict[str, float] | None:
    """Return exactly 32 finite source-image features, else abstain.

    All numbers derive ONLY from the eight validated shoulder/hip/knee/ankle
    joints. Coordinates and inclinations are 2D-camera-relative proxies.
    """
    if not np.isfinite(min_visibility) or not 0 <= min_visibility <= 1:
        raise ValueError("min_visibility must be finite in [0, 1]")
    try:
        if len(landmarks) != LANDMARK_COUNT:
            return None
        coords = np.asarray([
            [landmarks[i].x, landmarks[i].y, landmarks[i].z]
            for i in CRITICAL_JOINTS
        ], dtype=float)
        visibility = np.asarray([
            float(landmarks[i].visibility) for i in CRITICAL_JOINTS
        ], dtype=float)
    except (TypeError, ValueError, AttributeError, IndexError):
        return None
    if (coords.shape != (8, 3)
        or not np.isfinite(coords).all()
        or not np.isfinite(visibility).all()
        or (visibility < min_visibility).any()):
        return None
    points = dict(zip(CRITICAL_JOINTS, coords))
    width = float(np.linalg.norm(points[11] - points[12]))
    if width <= 1e-6:
        return None
    scale = width + 1e-6
    hips = (points[23] + points[24]) / 2.
    shoulders = (points[11] + points[12]) / 2.
    flattened = ((coords - hips) / scale).reshape(-1)
    left_trunk, right_trunk = points[11] - points[23], points[12] - points[24]
    extras = [
        calculate_angle(points[11], points[23], points[25]),
        calculate_angle(points[12], points[24], points[26]),
        calculate_angle(points[23], points[25], points[27]),
        calculate_angle(points[24], points[26], points[28]),
        float(np.degrees(np.arctan2(left_trunk[1], left_trunk[0]))),
        float(np.degrees(np.arctan2(right_trunk[1], right_trunk[0]))),
        float(np.linalg.norm(points[23] - points[24]) / scale),
        float((shoulders[0] - hips[0]) / scale),
    ]
    vector = np.r_[flattened, np.asarray(extras, dtype=float)]
    if vector.shape != (32,) or not np.isfinite(vector).all():
        return None
    return dict(zip(FEATURE_NAMES, map(float, vector)))
