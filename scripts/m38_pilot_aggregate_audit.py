"""M38: reproducible, aggregate-only research audit of the SAME M37 pilot.

No dataset/video/landmark access, no detector tuning and no claim of
statistical generalisation. This checks the three already-produced
M37, M37.1 and M37.2 JSON reports before further experiments.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CONSISTENT_KEYS = (
    "evaluation", "source_split", "source_mode", "annotation_sha256",
    "squat_candidate_videos", "squat_evaluated_videos", "excluded_videos",
    "source_total_annotated_reps", "m34_total_observed_cycles",
    "event_tp", "event_fp", "event_fn", "event_precision",
    "event_recall", "event_f1", "session_mean_absolute_count_error",
    "session_exact_count_fraction", "session_within_one_fraction",
    "zero_rep_negative_control_videos",
)


def _integer(value, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(label + " must be a non-negative integer")
    return value


def _ratio(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(label + " must be a finite numeric value")
    if not 0 <= value <= 1:
        raise ValueError(label + " must be between 0 and 1")
    return float(value)


def _same(a, b, label: str) -> None:
    if (type(a) in (int, float) and type(b) in (int, float) and
            not isinstance(a, bool) and not isinstance(b, bool)):
        if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError("Pilot report mismatch for " + label)
    elif a != b:
        raise ValueError("Pilot report mismatch for " + label)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON report object")
    return payload


def audit(pilot: dict, observability: dict, phase: dict) -> dict:
    """Strictly combine three distinct revisions of one frozen validation run."""
    for name, record in (("pilot", pilot), ("observability", observability),
                         ("phase", phase)):
        if not isinstance(record, dict):
            raise ValueError(name + " must be a JSON report object")
        if record.get("evaluation") != "m37-repcount-squat-event-v1":
            raise ValueError(name + " is not a recognized M37 evaluation")
        if record.get("source_split") != "validation":
            raise ValueError("Only previously examined validation pilot accepted")
        if record.get("source_mode") != "author_archive":
            raise ValueError("M38 requires the same author-archive source mode")
        sha = record.get("annotation_sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(
                char not in "0123456789abcdef" for char in sha):
            raise ValueError("Invalid annotation hash in " + name)
        n = _integer(record.get("squat_evaluated_videos"), name + " video count")
        candidates = _integer(record.get("squat_candidate_videos"),
                              name + " candidate count")
        excluded = _integer(record.get("excluded_videos"), name + " exclusions")
        if n < 1 or candidates != n + excluded:
            raise ValueError(name + " has inconsistent evaluation denominator")
        truth = _integer(record.get("source_total_annotated_reps"), "annotated reps")
        predicted = _integer(record.get("m34_total_observed_cycles"), "predicted reps")
        tp = _integer(record.get("event_tp"), "TP")
        fp = _integer(record.get("event_fp"), "FP")
        fn = _integer(record.get("event_fn"), "FN")
        if tp + fn != truth or tp + fp != predicted:
            raise ValueError(name + " event outcomes disagree with count totals")
        for field in ("session_exact_count_fraction",
                      "session_within_one_fraction"):
            _ratio(record.get(field), field)
        if _integer(record.get("zero_rep_negative_control_videos"),
                    "zero-rep clips") > n:
            raise ValueError("More zero-rep controls than evaluated clips")
        mae = record.get("session_mean_absolute_count_error")
        if type(mae) not in (int, float) or not math.isfinite(mae) or mae < 0:
            raise ValueError("Invalid count MAE")
        if n * mae + 1e-8 < abs(predicted - truth):
            raise ValueError("MAE violates the cohort triangle inequality")
        if predicted:
            _same(record.get("event_precision"), tp / predicted, "event precision")
        elif record.get("event_precision") is not None:
            raise ValueError("Precision must be null with no predicted events")
        if truth:
            _same(record.get("event_recall"), tp / truth, "event recall")
        elif record.get("event_recall") is not None:
            raise ValueError("Recall must be null with no true events")
        prec, rec = record.get("event_precision"), record.get("event_recall")
        expected_f1 = (2 * prec * rec / (prec + rec) if prec + rec else 0.0
                       ) if prec is not None and rec is not None else None
        _same(record.get("event_f1"), expected_f1, "event F1")
        if record.get("real_video_measured") is not True:
            raise ValueError("This audit requires owner-run real-video reports")
    for key in CONSISTENT_KEYS:
        if key not in pilot or key not in observability or key not in phase:
            raise ValueError("Missing required cohort field " + key)
        _same(pilot[key], observability[key], key)
        _same(pilot[key], phase[key], key)

    obs = observability.get("m37_observability")
    pw = phase.get("m37_phase_window_diagnostics")
    if not isinstance(obs, dict) or obs.get("protocol") != (
            "m37-squat-counter-observability-v1"):
        raise ValueError("Missing M37.1 observability protocol")
    if not isinstance(pw, dict) or pw.get("protocol") != (
            "m37-squat-phase-window-v1"):
        raise ValueError("Missing M37.2 phase-window protocol")
    if _integer(obs.get("videos_included"), "observable videos") != (
            pilot["squat_evaluated_videos"]):
        raise ValueError("M37.1 does not include every evaluated clip")
    if _integer(pw.get("videos_included"), "phase videos") != (
            pilot["squat_evaluated_videos"]):
        raise ValueError("M37.2 does not include every evaluated clip")
    okeys = (
        "input_frame_samples", "counter_accepted_ticks", "sampling_gate_skips",
        "knee_angle_usable_ticks", "knee_angle_unavailable_ticks",
        "knee_angle_high_ticks", "knee_angle_low_ticks", "knee_angle_middle_ticks",
        "confirmed_initial_high_transitions", "confirmed_low_transitions",
        "completed_return_transitions", "tracked_side_changes",
        "tracking_gap_resets",
    )
    pkeys = (
        "accepted_await_high", "accepted_await_low", "accepted_await_return",
        "high_while_await_high", "low_while_await_low", "high_while_await_return",
        "high_confirmation_broken", "low_confirmation_broken",
        "return_confirmation_broken", "side_switch_while_pending",
        "phase_expiration_compatible_resets", "annotated_cycles",
        "gt_cycles_accepted_lt4", "gt_cycles_accepted_lt6",
        "gt_cycles_usable_lt4", "gt_cycles_usable_lt6",
        "gt_cycles_no_usable_angle", "gt_cycles_with_high_and_low",
        "gt_cycles_with_low_then_high", "gt_cycles_with_two_high_and_two_low"
    )
    for key in okeys:
        _integer(obs.get(key), "M37.1 " + key)
    for key in pkeys:
        _integer(pw.get(key), "M37.2 " + key)
    if obs["counter_accepted_ticks"] + obs["sampling_gate_skips"] != (
            obs["input_frame_samples"]):
        raise ValueError("Invalid accepted vs skipped ticks")
    if obs["knee_angle_usable_ticks"] + obs["knee_angle_unavailable_ticks"] != (
            obs["counter_accepted_ticks"]):
        raise ValueError("Invalid accepted geometry coverage")
    if sum(obs[k] for k in ("knee_angle_high_ticks", "knee_angle_low_ticks",
                            "knee_angle_middle_ticks")) != obs["knee_angle_usable_ticks"]:
        raise ValueError("Invalid knee-angle bin totals")
    if sum(pw[k] for k in ("accepted_await_high", "accepted_await_low",
                           "accepted_await_return")) != obs["counter_accepted_ticks"]:
        raise ValueError("Inconsistent M37.1 vs M37.2 accepted ticks")
    if pw["annotated_cycles"] != pilot["source_total_annotated_reps"]:
        raise ValueError("Annotated window count differs from ground truth")
    if pw["side_switch_while_pending"] > obs["tracked_side_changes"]:
        raise ValueError("Pending side switches exceed observed side switches")
    for a, b in (("gt_cycles_accepted_lt4", "gt_cycles_accepted_lt6"),
                 ("gt_cycles_usable_lt4", "gt_cycles_usable_lt6"),
                 ("gt_cycles_with_low_then_high", "gt_cycles_with_high_and_low"),
                 ("gt_cycles_with_two_high_and_two_low",
                  "gt_cycles_with_high_and_low")):
        if pw[a] > pw[b]:
            raise ValueError("M37.2 cycle-window ordering inconsistent")
    if any(pw[k] > pw["annotated_cycles"] for k in pkeys
           if k.startswith("gt_cycles_")):
        raise ValueError("M37.2 window counts exceed annotated cycles")

    n = pilot["squat_evaluated_videos"]
    positives = pilot["source_total_annotated_reps"]
    baseline_mae = positives / n  # All publisher counts are >= 0.
    m34_mae = pilot["session_mean_absolute_count_error"]
    return {
        "protocol": "m38-m37-pilot-aggregate-baseline-audit-v1",
        "study_type": "post_hoc_exploratory_validation_failure_analysis",
        "source_split": "validation",
        "annotation_sha256": pilot["annotation_sha256"],
        "videos": n,
        "annotated_reps": positives,
        "m34": {
            "predicted_reps": pilot["m34_total_observed_cycles"],
            "event_tp": pilot["event_tp"], "event_fp": pilot["event_fp"],
            "event_fn": pilot["event_fn"],
            "event_precision": pilot["event_precision"],
            "event_recall": pilot["event_recall"],
            "event_f1": pilot["event_f1"],
            "video_count_mae": m34_mae
        },
        "always_zero_count_baseline": {
            "predicted_reps": 0,
            "video_count_mae": baseline_mae,
            "event_recall": 0.0 if positives else None,
            "event_precision": None,
            "event_f1": None,
            "note": "Undefined event precision/F1 (zero predicted positives)."
        },
        "m34_mae_improvement_vs_always_zero_reps_per_video":
            baseline_mae - m34_mae,
        "observability": {
            "input_frame_samples": obs["input_frame_samples"],
            "accepted_ticks": obs["counter_accepted_ticks"],
            "sampling_gate_skips": obs["sampling_gate_skips"],
            "usable_angle_ticks": obs["knee_angle_usable_ticks"],
            "fraction_accepted_ticks_usable": (
                obs["knee_angle_usable_ticks"] / obs["counter_accepted_ticks"]
                if obs["counter_accepted_ticks"] else None
            ),
            "high_ticks": obs["knee_angle_high_ticks"],
            "low_ticks": obs["knee_angle_low_ticks"],
            "low_confirmation_broken": pw["low_confirmation_broken"],
            "side_switch_while_pending": pw["side_switch_while_pending"],
            "phase_expiration_compatible_resets":
                pw["phase_expiration_compatible_resets"]
        },
        "annotated_window_evidence": {
            "cycles_with_less_than_6_usable_ticks": pw["gt_cycles_usable_lt6"],
            "fraction_with_less_than_6_usable_ticks": (
                pw["gt_cycles_usable_lt6"] / positives if positives else None
            ),
            "cycles_with_less_than_4_usable_ticks": pw["gt_cycles_usable_lt4"],
            "cycles_with_two_high_and_two_low":
                pw["gt_cycles_with_two_high_and_two_low"],
            "cycles_with_low_then_high": pw["gt_cycles_with_low_then_high"]
        },
        "evidence_limits": [
            "Only three videos; no independent subject-level uncertainty estimate.",
            "Original validation clips have now been inspected post hoc.",
            "Half-open publisher cycle intervals may omit required high endpoints.",
            "Causal attribution and frame/annotation parity are unverified.",
            "MAE baseline improvement is descriptive, not a paired significance test.",
            "No rep-quality, injury, or browser-estimator accuracy conclusion."
        ],
        "next_step": (
            "Freeze original M34 and validation evidence. Design new candidate "
            "detectors only on a separate publisher training/development split; "
            "pre-register independent test selection and event-matching policy."
        ),
        "raw_video_or_landmark_accessed": False,
        "m34_counter_modified": False
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--observability", required=True, type=Path)
    parser.add_argument("--phase", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        # No output file can be overwritten; read only three selected JSONs.
        if args.report.exists():
            raise FileExistsError("M38 report exists; refusing overwrite")
        output = audit(_read_json(args.pilot), _read_json(args.observability),
                       _read_json(args.phase))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(output, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError, KeyError,
            json.JSONDecodeError) as error:
        parser.exit(1, "M38 stopped: " + str(error) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
