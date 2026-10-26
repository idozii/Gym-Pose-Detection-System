"""Headless regression tests: report WHY 0/120 webcam tasks abstained.

No owner files, images, original TEST content or real camera used.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gym_pose.live_diagnostics import (
    LiveSessionDiagnostics, source_gate_diagnostic,
)


def landmarks(*, invisible=(), nan_joint=None):
    values = [
        SimpleNamespace(x=.02 * i, y=.01 * (i + i % 3), z=.001 * i,
                        visibility=(.1 if i in invisible else .99))
        for i in range(33)
    ]
    if nan_joint is not None:
        values[nan_joint].x = float("nan")
    return values


def test_squat_no_pose_and_invisible_knee_ankle_are_distinguishable():
    assert source_gate_diagnostic(
        None, "squat", "error_knees_forward") == ("no_pose_detected", ())
    reason, missing = source_gate_diagnostic(
        landmarks(invisible=(25, 28)), "squat", "error_knees_forward")
    assert reason == "required_joint_low_visibility"
    assert missing == ("left_knee", "right_ankle")
    assert source_gate_diagnostic(
        landmarks(nan_joint=0), "squat", "error_knees_forward")[0] == (
            "nonfinite_landmark_values")
    assert source_gate_diagnostic(
        landmarks()[:32], "squat", "error_knees_forward")[0] == (
            "invalid_landmark_count")


def test_ohp_error_adapters_have_different_required_joint_visibility():
    low_ankles = landmarks(invisible=(27, 28))
    assert source_gate_diagnostic(
        low_ankles, "ohp", "error_elbows")[0] == (
            "passes_basic_visibility_geometry_checks")
    assert source_gate_diagnostic(
        low_ankles, "ohp", "error_knees")[0] == (
            "required_joint_low_visibility")
    low_wrists = landmarks(invisible=(15, 16))
    assert source_gate_diagnostic(
        low_wrists, "ohp", "error_elbows")[0] == (
            "required_joint_low_visibility")
    assert source_gate_diagnostic(
        low_wrists, "ohp", "error_knees")[0] == (
            "passes_basic_visibility_geometry_checks")


def test_row_does_not_require_legs_and_invalid_geometry_is_separate():
    assert source_gate_diagnostic(
        landmarks(invisible=(25, 26, 27, 28)),
        "barbell_row", "lumbar")[0] == (
            "passes_basic_visibility_geometry_checks")
    assert source_gate_diagnostic(
        landmarks(invisible=(11,)),
        "barbell_row", "torso_angle")[0] == (
            "required_joint_low_visibility")
    degenerate = landmarks()
    degenerate[12].x = degenerate[11].x
    degenerate[12].y = degenerate[11].y
    degenerate[12].z = degenerate[11].z
    assert source_gate_diagnostic(
        degenerate, "barbell_row", "lumbar")[0] == "degenerate_shoulder_width"
    with pytest.raises(ValueError, match="Unknown viewer target"):
        source_gate_diagnostic(landmarks(), "deadlift", "wrong")


def _abstain(reason="missing_or_rejected_pose"):
    return {"status": "abstain", "reason": reason,
            "source_label_flag": None}


def test_zero_scores_from_60_no_pose_frames_explains_120_abstentions():
    info = LiveSessionDiagnostics()
    for _ in range(60):
        info.record(
            "squat", None, {
                "error_knees_forward": _abstain(),
                "error_knees_inward": _abstain(),
            }, pose_ms=8, model_ms=.1, frame_ms=22)
    out = info.summary(elapsed_seconds=2)
    assert out["frames_processed"] == 60
    assert out["no_pose_frames"] == 60
    assert out["pose_detected_frames"] == 0
    assert out["pose_detection_rate"] == 0
    assert out["abstentions_by_target_and_reason"] == {
        "squat/error_knees_forward": {"no_pose_detected": 60},
        "squat/error_knees_inward": {"no_pose_detected": 60},
    }
    assert out["observed_fps_including_display"] == 30
    assert out["timing_ms"]["pose_process"]["mean"] == 8
    assert out["no_frame_images_or_per_frame_data_saved"] is True


def test_low_visibility_counted_once_per_task_and_probability_not_persisted():
    info = LiveSessionDiagnostics()
    for _ in range(3):
        info.record(
            "squat", landmarks(invisible=(27,)), {
                "error_knees_forward": _abstain(),
                "error_knees_inward": _abstain(),
            }, pose_ms=9, model_ms=2, frame_ms=25)
    info.record(
        "squat", landmarks(), {
            "error_knees_forward": {
                "status": "experimental_source_label_score",
                "error_probability": .89, "source_label_flag": True,
            },
            "error_knees_inward": {
                "status": "experimental_source_label_score",
                "error_probability": .02, "source_label_flag": False,
            },
        }, pose_ms=12, model_ms=2, frame_ms=30)
    out = info.summary(elapsed_seconds=1)
    assert out["pose_detected_frames"] == 4
    assert out["no_pose_frames"] == 0
    assert out["scored_by_target"]["squat/error_knees_forward"] == 1
    assert out["scored_by_target"]["squat/error_knees_inward"] == 1
    assert out["abstentions_by_target_and_reason"][
        "squat/error_knees_forward"]["required_joint_low_visibility"] == 3
    assert out["required_low_visibility_joint_occurrences"][
        "squat/error_knees_inward"]["left_ankle"] == 3
    assert "0.89" not in str(out) and "0.02" not in str(out)
    assert "landmarks" not in str(out).lower()


def test_unloaded_model_and_model_exception_are_not_mislabeled_pose_failure():
    info = LiveSessionDiagnostics()
    info.record(
        "ohp", landmarks(), {
            "error_elbows": _abstain("model_not_loaded_or_unavailable"),
            "error_knees": _abstain("model_inference_error"),
        }, pose_ms=1, model_ms=1, frame_ms=5)
    result = info.summary(elapsed_seconds=1)
    assert result["abstentions_by_target_and_reason"] == {
        "ohp/error_elbows": {"model_not_loaded_or_unavailable": 1},
        "ohp/error_knees": {"model_inference_error": 1},
    }
    with pytest.raises(ValueError, match="timings"):
        info.record("ohp", None, {}, pose_ms=-1,
                    model_ms=0, frame_ms=0)
