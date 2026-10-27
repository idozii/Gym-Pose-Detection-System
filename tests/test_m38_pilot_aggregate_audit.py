"""M38 aggregate audit: no real RepCount bytes or user files needed."""
from __future__ import annotations

import json
import pytest

from scripts.m38_pilot_aggregate_audit import audit, main


def reports():
    shared = {
        "evaluation": "m37-repcount-squat-event-v1",
        "source_split": "validation", "source_mode": "author_archive",
        "annotation_sha256": "9" * 64,
        "squat_candidate_videos": 3, "squat_evaluated_videos": 3,
        "excluded_videos": 0, "source_total_annotated_reps": 45,
        "m34_total_observed_cycles": 1, "event_tp": 1,
        "event_fp": 0, "event_fn": 44, "event_precision": 1.0,
        "event_recall": 1 / 45, "event_f1": 2 / 46,
        "session_mean_absolute_count_error": 44 / 3,
        "session_exact_count_fraction": 0.,
        "session_within_one_fraction": 0.,
        "zero_rep_negative_control_videos": 0,
        "real_video_measured": True,
    }
    obs = {
        "protocol": "m37-squat-counter-observability-v1",
        "videos_included": 3,
        "input_frame_samples": 210, "counter_accepted_ticks": 210,
        "sampling_gate_skips": 0, "knee_angle_usable_ticks": 185,
        "knee_angle_unavailable_ticks": 25,
        "knee_angle_high_ticks": 80, "knee_angle_low_ticks": 56,
        "knee_angle_middle_ticks": 49, "confirmed_initial_high_transitions": 5,
        "confirmed_low_transitions": 2, "completed_return_transitions": 1,
        "tracked_side_changes": 3, "tracking_gap_resets": 0,
    }
    phase = {
        "protocol": "m37-squat-phase-window-v1",
        "videos_included": 3,
        "accepted_await_high": 92, "accepted_await_low": 114,
        "accepted_await_return": 4,
        "high_while_await_high": 16, "low_while_await_low": 25,
        "high_while_await_return": 2, "high_confirmation_broken": 6,
        "low_confirmation_broken": 18,
        "return_confirmation_broken": 0,
        "side_switch_while_pending": 3,
        "phase_expiration_compatible_resets": 2,
        "annotated_cycles": 45, "gt_cycles_accepted_lt4": 29,
        "gt_cycles_accepted_lt6": 43, "gt_cycles_usable_lt4": 31,
        "gt_cycles_usable_lt6": 44, "gt_cycles_no_usable_angle": 2,
        "gt_cycles_with_high_and_low": 27,
        "gt_cycles_with_low_then_high": 16,
        "gt_cycles_with_two_high_and_two_low": 3
    }
    return dict(shared), {**shared, "m37_observability": obs}, {
        **shared, "m37_phase_window_diagnostics": phase
    }


def test_m38_exact_real_pilot_numbers_are_synthetic_fixtures_not_real_data():
    pilot, obs, phase = reports()
    result = audit(pilot, obs, phase)
    assert result["videos"] == 3
    assert result["annotated_reps"] == 45
    assert result["m34"]["event_fn"] == 44
    assert result["m34"]["event_f1"] == pytest.approx(2 / 46)
    assert result["always_zero_count_baseline"]["video_count_mae"] == 15
    assert result["always_zero_count_baseline"]["event_f1"] is None
    assert result["m34_mae_improvement_vs_always_zero_reps_per_video"] == pytest.approx(1 / 3)
    assert result["observability"]["fraction_accepted_ticks_usable"] == pytest.approx(185 / 210)
    assert result["annotated_window_evidence"]["fraction_with_less_than_6_usable_ticks"] == pytest.approx(44 / 45)
    assert result["raw_video_or_landmark_accessed"] is False
    assert result["m34_counter_modified"] is False
    output = json.dumps(result)
    assert "video_path" not in output
    assert "landmarks" not in output
    assert "predicted_event_times" not in output
    assert "stu9_" not in output


@pytest.mark.parametrize("which,key,value,message", [
    ("obs", "annotation_sha256", "a" * 64, "mismatch"),
    ("phase", "event_fn", 43, "disagree|mismatch"),
    ("pilot", "source_split", "test", "validation"),
    ("pilot", "squat_evaluated_videos", 0, "denominator"),
    ("pilot", "event_tp", 2, "disagree"),
    ("pilot", "session_mean_absolute_count_error", 0, "triangle"),
])
def test_m38_rejects_stale_or_inconsistent_run(which, key, value, message):
    pilot, obs, phase = reports()
    items = {"pilot": pilot, "obs": obs, "phase": phase}
    items[which][key] = value
    with pytest.raises(ValueError, match=message):
        audit(pilot, obs, phase)


def test_m38_rejects_invalid_phase_cross_check():
    pilot, obs, phase = reports()
    phase["m37_phase_window_diagnostics"]["accepted_await_high"] = 93
    with pytest.raises(ValueError, match="accepted ticks"):
        audit(pilot, obs, phase)


def test_m38_rejects_missing_observability_without_fabricating_failures():
    pilot, obs, phase = reports()
    obs.pop("m37_observability")
    with pytest.raises(ValueError, match="M37.1"):
        audit(pilot, obs, phase)


def test_m38_cli_writes_only_new_aggregate_json(tmp_path, capsys):
    pilot, obs, phase = reports()
    sources = []
    for name, payload in (("pilot", pilot), ("obs", obs), ("phase", phase)):
        p = tmp_path / (name + ".json")
        p.write_text(json.dumps(payload), encoding="utf-8")
        sources.append(p)
    dest = tmp_path / "report.json"
    args = [
        "--pilot", str(sources[0]), "--observability", str(sources[1]),
        "--phase", str(sources[2]), "--report", str(dest)
    ]
    assert main(args) == 0
    assert json.loads(dest.read_text())["annotated_reps"] == 45
    assert json.loads(capsys.readouterr().out)["videos"] == 3
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 1
