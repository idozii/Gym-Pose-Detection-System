"""Explain why the *existing* OHP 111-feature quality gate rejects a pose.

Diagnostic only: production thresholds, schemas and accepted frame CSVs stay
unchanged. First failure reason is exclusive, while failed-joint counts may
overlap across categories. Camera-relative estimates cannot validate exercise
technique or joint biomechanics.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .features import IMPORTANT_INDICES, LANDMARK_COUNT
from .ohp_features import (
    OHP_ADDITIONAL_CRITICAL_INDICES, pose_to_ohp_features,
)
from .quality import JOINT_NAMES as BASE_JOINT_NAMES

JOINT_NAMES = {
    **BASE_JOINT_NAMES,
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
}
REASONS = (
    "accepted", "no_pose_detected", "invalid_landmark_structure",
    "nonfinite_coordinates_or_visibility",
    "squat_base_joint_low_visibility",
    "ohp_upper_joint_low_visibility",
    "degenerate_shoulder_distance", "other_feature_rejection",
)
OHP_CRITICAL_INDICES = (*IMPORTANT_INDICES, *OHP_ADDITIONAL_CRITICAL_INDICES)


def assess_ohp_pose_quality(landmarks: Any, *,
                            min_visibility: float = .5) -> dict:
    if not 0 <= min_visibility <= 1:
        raise ValueError("min_visibility must be within [0,1]")
    result = {
        "reason": "accepted", "low_visibility_joints": [],
        "low_visibility_base_joints": [], "low_visibility_upper_joints": [],
        "minimum_critical_joint_visibility": None,
    }
    if landmarks is None:
        result["reason"] = "no_pose_detected"
        return result
    try:
        if len(landmarks) != LANDMARK_COUNT:
            result["reason"] = "invalid_landmark_structure"
            return result
        coords = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
        visibility = np.asarray([
            float(getattr(p, "visibility", 0.0)) for p in landmarks
        ], dtype=float)
    except (ValueError, TypeError, AttributeError, IndexError):
        result["reason"] = "invalid_landmark_structure"
        return result
    if (coords.shape != (LANDMARK_COUNT, 3)
        or not np.isfinite(coords).all() or not np.isfinite(visibility).all()):
        result["reason"] = "nonfinite_coordinates_or_visibility"
        return result
    key_vis = visibility[list(OHP_CRITICAL_INDICES)]
    result["minimum_critical_joint_visibility"] = float(key_vis.min())
    result["low_visibility_base_joints"] = [
        JOINT_NAMES[i] for i in IMPORTANT_INDICES if visibility[i] < min_visibility
    ]
    result["low_visibility_upper_joints"] = [
        JOINT_NAMES[i] for i in OHP_ADDITIONAL_CRITICAL_INDICES
        if visibility[i] < min_visibility
    ]
    result["low_visibility_joints"] = (
        result["low_visibility_base_joints"]
        + result["low_visibility_upper_joints"]
    )
    # Mirror OHP feature extraction ordering: the old Squat 103-feature
    # quality gate runs first, then upper-limb elbow/wrist visibility.
    if result["low_visibility_base_joints"]:
        result["reason"] = "squat_base_joint_low_visibility"
    elif result["low_visibility_upper_joints"]:
        result["reason"] = "ohp_upper_joint_low_visibility"
    elif float(np.linalg.norm(coords[11] - coords[12])) <= 1e-6:
        result["reason"] = "degenerate_shoulder_distance"
    elif pose_to_ohp_features(landmarks, min_visibility=min_visibility) is None:
        result["reason"] = "other_feature_rejection"
    return result

# Pure visibility-admission hypotheses. NO existing model accepts these
# task-specific masks; they are feasibility counts only, not inference rules.
#
# Elbow geometry uses shoulders, elbows and wrists. Knee geometry uses
# shoulders (normalization), hips, knees and ankles. Source labels remain
# unchanged. The original full 111-feature RF schema still assumes the
# original production all-joint gate; task-gate training must be redesigned
# and separately validated before any deployment.
OHP_ARM_CHAIN = (11, 12, 13, 14, 15, 16)
OHP_KNEE_CHAIN = (11, 12, 23, 24, 25, 26, 27, 28)
TASK_GATE_POLICIES = (
    "all_050",
    "all_035",
    "elbows_arm_chain_050",
    "elbows_arm_050_other_035",
    "knees_lower_chain_050",
    "knees_lower_050_upper_035",
    "all_except_ankles_050",
)


def assess_task_gate_policies(landmarks: Any, *,
                              strict: float = .5, relaxed: float = .35) -> dict[str, bool]:
    """Admission-only counterfactuals computed from the SAME detected frame.

    All policies reject missing/malformed/nonfinite poses or degenerate
    shoulders. This function never imputes coordinates or emits model features.
    """
    if not (0 <= relaxed <= strict <= 1):
        raise ValueError("Require 0 <= relaxed <= strict <= 1")
    denied = dict.fromkeys(TASK_GATE_POLICIES, False)
    if landmarks is None:
        return denied
    try:
        if len(landmarks) != LANDMARK_COUNT:
            return denied
        coords = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
        visibility = np.asarray([
            float(getattr(p, "visibility", 0.0)) for p in landmarks
        ], dtype=float)
    except (ValueError, TypeError, AttributeError, IndexError):
        return denied
    if (coords.shape != (LANDMARK_COUNT, 3)
        or not np.isfinite(coords).all() or not np.isfinite(visibility).all()
        or float(np.linalg.norm(coords[11] - coords[12])) <= 1e-6):
        return denied

    base = tuple(IMPORTANT_INDICES)
    arms = tuple(OHP_ADDITIONAL_CRITICAL_INDICES)
    full = (*base, *arms)
    def meets(ids, minimum):
        return all(visibility[i] >= minimum for i in ids)

    return {
        "all_050": meets(full, strict),
        "all_035": meets(full, relaxed),
        "elbows_arm_chain_050": meets(OHP_ARM_CHAIN, strict),
        "elbows_arm_050_other_035": (
            meets(arms, strict) and meets(base, relaxed)
        ),
        "knees_lower_chain_050": meets(OHP_KNEE_CHAIN, strict),
        "knees_lower_050_upper_035": (
            meets(base, strict) and meets(arms, relaxed)
        ),
        "all_except_ankles_050": meets(
            tuple(i for i in full if i not in (27,28)), strict
        ),
    }
