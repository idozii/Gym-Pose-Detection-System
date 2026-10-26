"""M16D headless, recording-free task latency profile regressions.

Tests represent the user-observed 47/180 admitted Squat frames and ensure
reported RF latency is measured only for actual successful model calls.
No physical webcam, publisher TEST or owner-specific pickle ever accessed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gym_pose.live_diagnostics import LiveSessionDiagnostics
from gym_pose.research_viewer import analyze_landmarks


def points(*, bad_ankle=False):
    return [
        SimpleNamespace(x=.01*i, y=.03*i, z=0.,
                        visibility=.1 if bad_ankle and i == 28 else .99)
        for i in range(33)
    ]


def result(status):
    if status == "experimental_source_label_score":
        return {
            "status": status, "error_probability": .79,
            "threshold": .3, "source_label_flag": True,
        }
    return {"status": "abstain", "reason": "missing_or_rejected_pose",
            "source_label_flag": None}


def test_scored_only_latency_matches_47_of_180_real_webcam_pattern():
    diagnostics = LiveSessionDiagnostics()
    for i in range(180):
        admitted = i < 47
        landmark = points(bad_ankle=not admitted)
        reports = {
            "error_knees_forward": result(
                "experimental_source_label_score" if admitted else "abstain"),
            "error_knees_inward": result(
                "experimental_source_label_score" if admitted else "abstain"),
        }
        for task, response in reports.items():
            diagnostics.record_task_timing(
                "squat", task, response["status"],
                12.5 if admitted else .2)
        diagnostics.record(
            "squat", landmark, reports,
            pose_ms=25., model_ms=26. if admitted else .4,
            frame_ms=70. if admitted else 60.)
    summary = diagnostics.summary(elapsed_seconds=180/15.8)
    assert summary["frames_processed"] == 180
    assert summary["pose_detected_frames"] == 180
    assert sum(summary["scored_by_target"].values()) == 94
    assert summary["timed_successful_task_calls_in_window"] == 94
    assert summary["timed_abstained_task_calls_in_window"] == 266
    for task in ("error_knees_forward", "error_knees_inward"):
        row = summary["source_label_score_admission_by_target"][f"squat/{task}"]
        assert row["source_frames_seen"] == 180
        assert row["scores_emitted"] == 47
        assert row["abstentions"] == 133
        assert row["score_admission_rate"] == .2611
    timings = summary["timing_ms"]
    assert timings["successful_model_task_inference"]["mean"] == 12.5
    assert timings["abstained_model_task_call"]["mean"] == .2
    assert timings["model_scoring"]["mean"] < 12.5
    assert summary["observed_fps_including_display"] == 15.8
    assert "0.79" not in str(summary)
    assert "source_label_flag" not in str(summary)


def test_mixed_ohp_frame_reports_distinct_target_timing():
    diagnostics = LiveSessionDiagnostics()
    diagnostics.record_task_timing(
        "ohp", "error_elbows", "experimental_source_label_score", 21.)
    diagnostics.record_task_timing("ohp", "error_knees", "abstain", .1)
    diagnostics.record(
        "ohp", points(bad_ankle=True), {
            "error_elbows": result("experimental_source_label_score"),
            "error_knees": result("abstain"),
        }, pose_ms=10., model_ms=21.1, frame_ms=45.)
    out = diagnostics.summary(elapsed_seconds=1)
    assert out["source_label_score_admission_by_target"]["ohp/error_elbows"][
        "score_admission_rate"] == 1.
    assert out["source_label_score_admission_by_target"]["ohp/error_knees"][
        "score_admission_rate"] == 0.
    assert out["timing_ms"]["by_target_successful_task_inference"][
        "ohp/error_elbows"]["mean"] == 21.
    assert out["timing_ms"]["by_target_abstained_task_call"][
        "ohp/error_knees"]["mean"] == .1


def test_timing_callback_measures_exactly_two_source_task_calls():
    class StubRegistry:
        def score_landmarks(self, exercise, target, landmarks, **kwargs):
            if target == "error_knees_inward":
                raise RuntimeError("An inference exception is treated as abstain")
            return result("experimental_source_label_score")
    times = []
    scored = analyze_landmarks(
        StubRegistry(), "squat", points(),
        on_task_timing=lambda *values: times.append(values))
    assert len(times) == 2
    assert times[0][:3] == (
        "squat", "error_knees_forward", "experimental_source_label_score")
    assert times[1][:3] == ("squat", "error_knees_inward", "abstain")
    assert all(isinstance(record[-1], float) and record[-1] >= 0 for record in times)
    assert scored["error_knees_inward"]["source_label_flag"] is None


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf"), "8ms"])
def test_invalid_task_duration_fails_closed(value):
    d = LiveSessionDiagnostics()
    with pytest.raises(ValueError):
        d.record_task_timing("squat", "error_knees_forward",
                             "experimental_source_label_score", value)


def test_unknown_task_or_unknown_status_is_rejected():
    d = LiveSessionDiagnostics()
    with pytest.raises(ValueError, match="Unknown source-label"):
        d.record_task_timing("unknown", "unknown", "abstain", 1.)
    with pytest.raises(ValueError, match="Unknown source-label prediction"):
        d.record_task_timing(
            "squat", "error_knees_forward", "correct_form", 1.)


def test_no_scored_task_is_explicitly_missing_not_zero_latency():
    d = LiveSessionDiagnostics()
    d.record_task_timing("squat", "error_knees_forward", "abstain", .1)
    d.record(
        "squat", None, {
            "error_knees_forward": result("abstain"),
            "error_knees_inward": result("abstain"),
        }, pose_ms=1., model_ms=.2, frame_ms=10.)
    report = d.summary(elapsed_seconds=.1)
    assert report["timing_ms"]["successful_model_task_inference"] == {
        "mean": None, "p95": None, "max": None,
    }
    assert report["timed_successful_task_calls_in_window"] == 0
    assert report["source_label_score_admission_by_target"][
        "squat/error_knees_forward"]["score_admission_rate"] == 0.
