"""OHP-specific pose extension. This is a camera-relative engineering baseline.

Keeps the original versioned 103-column feature schema untouched and adds
eight separately versioned OHP features. These angles/ratios are NOT physical
or clinical measurement of pressing form. MediaPipe joint visibility is
required for elbows and wrists before accepting an OHP pose.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .features import (
    FEATURE_NAMES, LANDMARK_COUNT, calculate_angle, pose_to_features,
)

OHP_SCHEMA_VERSION = "ohp-pose-111-v1"
OHP_EXTRA_NAMES = (
    "left_elbow_angle_deg", "right_elbow_angle_deg",
    "left_knee_angle_deg", "right_knee_angle_deg",
    "left_wrist_minus_shoulder_y_norm", "right_wrist_minus_shoulder_y_norm",
    "elbow_width_x_over_shoulder_dist", "wrist_width_x_over_shoulder_dist",
)
OHP_FEATURE_NAMES = (*FEATURE_NAMES, *OHP_EXTRA_NAMES)
OHP_ADDITIONAL_CRITICAL_INDICES = (13, 14, 15, 16)


def pose_to_ohp_features(landmarks: Any, *,
                         min_visibility: float = .5) -> dict[str, float] | None:
    """Return strictly named 111 columns or None when key joints are unreliable."""
    if not 0 <= min_visibility <= 1:
        raise ValueError("min_visibility must be within [0,1]")
    base = pose_to_features(landmarks, min_visibility=min_visibility)
    if base is None:
        return None
    try:
        if len(landmarks) != LANDMARK_COUNT:
            return None
        visibility = np.asarray([
            float(getattr(landmarks[i], "visibility", 0.0))
            for i in OHP_ADDITIONAL_CRITICAL_INDICES
        ])
        if not np.isfinite(visibility).all() or (visibility < min_visibility).any():
            return None
        coords = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=float)
        if coords.shape != (LANDMARK_COUNT, 3) or not np.isfinite(coords).all():
            return None
        shoulder_width = float(np.linalg.norm(coords[11] - coords[12]))
        if shoulder_width <= 1e-6:
            return None
        # Coordinates are camera-relative; normalisation agrees with legacy 103.
        extras = (
            calculate_angle(coords[11], coords[13], coords[15]),
            calculate_angle(coords[12], coords[14], coords[16]),
            calculate_angle(coords[23], coords[25], coords[27]),
            calculate_angle(coords[24], coords[26], coords[28]),
            (coords[15, 1] - coords[11, 1]) / shoulder_width,
            (coords[16, 1] - coords[12, 1]) / shoulder_width,
            abs(coords[13, 0] - coords[14, 0]) / shoulder_width,
            abs(coords[15, 0] - coords[16, 0]) / shoulder_width,
        )
        if len(extras) != 8 or not np.isfinite(extras).all():
            return None
        return {**base, **dict(zip(OHP_EXTRA_NAMES, map(float, extras)))}
    except (TypeError, ValueError, AttributeError, IndexError):
        return None
