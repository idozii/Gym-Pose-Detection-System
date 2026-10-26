"""Task-specific, camera-relative OHP pose features with trustworthy input joints.

These are deliberately separate from legacy OHP 111-feature models.
An elbow detector cannot use unseen knee/ankle coordinates as trusted inputs;
a knee detector must not require visible wrists/elbows. Neither schema is a
clinical measurement of exercise correctness or injury risk.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .features import LANDMARK_COUNT, calculate_angle

JOINT_NAMES = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
UPPER8 = (11, 12, 13, 14, 15, 16, 23, 24)
LOWER8 = (11, 12, 23, 24, 25, 26, 27, 28)
TASKS = ("error_elbows", "error_knees")
JOINTS_BY_TASK = {"error_elbows": UPPER8, "error_knees": LOWER8}
SCHEMA_BY_TASK = {
    "error_elbows": "ohp-elbows-upper8-30-v1",
    "error_knees": "ohp-knees-lower8-30-v1",
}
EXTRAS_BY_TASK = {
    "error_elbows": (
        "left_elbow_angle_deg", "right_elbow_angle_deg",
        "left_wrist_minus_shoulder_y_norm", "right_wrist_minus_shoulder_y_norm",
        "elbow_width_x_over_shoulder_dist", "wrist_width_x_over_shoulder_dist",
    ),
    "error_knees": (
        "left_knee_angle_deg", "right_knee_angle_deg",
        "left_hip_angle_deg", "right_hip_angle_deg",
        "torso_incline_deg", "hip_to_shoulder_ratio",
    ),
}
FEATURES_BY_TASK = {
    task: tuple(f"lm_{index}_{axis}"
                for index in JOINTS_BY_TASK[task] for axis in ("x", "y", "z"))
    + EXTRAS_BY_TASK[task]
    for task in TASKS
}


def pose_to_task_features(landmarks: Any, *, task: str,
                          min_visibility: float = .5) -> dict[str, float] | None:
    """30 features from only the eight geometrically required, visible joints.

    Hip-centre/shoulder-distance normalization is consistent with the existing
    MediaPipe feature convention. Crucially, low-visibility joints not used
    by this specific model do not enter its input vector.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown OHP error task: {task}")
    if not np.isfinite(min_visibility) or not 0 <= min_visibility <= 1:
        raise ValueError("min_visibility must be finite and in [0,1]")
    selected = JOINTS_BY_TASK[task]
    try:
        if len(landmarks) != LANDMARK_COUNT:
            return None
        coords = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z]
                           for i in selected], dtype=np.float32)
        vis = np.array([float(landmarks[i].visibility) for i in selected],
                       dtype=float)
    except (TypeError, ValueError, AttributeError, IndexError):
        return None
    if (coords.shape != (8, 3) or not np.isfinite(coords).all()
        or not np.isfinite(vis).all() or (vis < min_visibility).any()):
        return None
    point = {key: coords[pos] for pos, key in enumerate(selected)}
    hip_centre = (point[23] + point[24]) / 2.
    distance = float(np.linalg.norm(point[11] - point[12]))
    if not np.isfinite(distance) or distance <= 1e-6:
        return None
    scale = distance + 1e-6
    norm = ((coords - hip_centre) / scale).reshape(-1)

    if task == "error_elbows":
        extras = (
            calculate_angle(point[11], point[13], point[15]),
            calculate_angle(point[12], point[14], point[16]),
            (point[15][1] - point[11][1]) / scale,
            (point[16][1] - point[12][1]) / scale,
            abs(point[13][0] - point[14][0]) / scale,
            abs(point[15][0] - point[16][0]) / scale,
        )
    else:
        torso = point[11] - point[23]
        extras = (
            calculate_angle(point[23], point[25], point[27]),
            calculate_angle(point[24], point[26], point[28]),
            calculate_angle(point[11], point[23], point[25]),
            calculate_angle(point[12], point[24], point[26]),
            float(np.degrees(np.arctan2(torso[1], torso[0]))),
            np.linalg.norm(point[23] - point[24]) / scale,
        )
    values = np.r_[norm, np.asarray(extras, dtype=float)]
    if values.shape != (30,) or not np.isfinite(values).all():
        return None
    return dict(zip(FEATURES_BY_TASK[task], map(float, values)))
