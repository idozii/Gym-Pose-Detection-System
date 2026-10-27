"""M43 minimal, strict aggregate schema for exploratory TRAIN error phenotypes.

The frozen JavaScript replay produces per-video temporary summaries.
Python checks all partitions and persists ONLY cohort-level counts.
No labels are presented as causes of a missed repetition.
"""
from __future__ import annotations

from scripts.m41_repcount_train_cadence import CONDITIONS

PROTOCOL = "m43-train-temporal-error-audit-v1"

FN_KEYS = (
    "no_usable_angle_in_gt_window",
    "usable_but_no_low_in_gt_window",
    "low_but_no_later_high_in_gt_window",
    "low_then_high_in_gt_window_still_unmatched",
)
FP_KEYS = (
    "unmatched_within_some_gt_end_tolerance",
    "unmatched_outside_all_gt_end_tolerances",
)
WINDOW_KEYS = (
    "all_gt_windows_usable_lt4",
    "all_gt_windows_usable_lt6",
    "all_gt_windows_no_usable",
    "all_gt_windows_with_any_low",
    "all_gt_windows_with_low_then_high",
    "side_switch_resets_while_pending",
    "tracking_gap_resets_while_pending",
)
PAIRED_KEYS = (
    "both_matched", "dense_only_matched",
    "subsample_only_matched", "neither_matched",
)


def _nonnegative(value, label):
    if type(value) is not int or value < 0:
        raise ValueError("M43 invalid diagnostic counter " + label)
    return value


def _counts(record, keys, name):
    if not isinstance(record, dict) or set(record) != set(keys):
        raise ValueError("M43 invalid diagnostic bucket schema " + name)
    return {key: _nonnegative(record[key], name + ":" + key) for key in keys}


def validate_m43_diagnostics(data: dict, m41: dict) -> None:
    """Refuse any synthetic or malformed event partition inconsistent with M41."""
    if not isinstance(data, dict):
        raise ValueError("M43 diagnostics missing")
    if set(data) != {"dense_200ms", "subsample_400ms", "paired_gt_outcomes"}:
        raise ValueError("M43 diagnostics conditions mismatch")
    ground_truth = _nonnegative(m41.get("annotated_rep_count"), "GT count")
    conditions = {}
    for condition in CONDITIONS:
        d = data[condition]
        if not isinstance(d, dict) or set(d) != {
            "gt_events", "tp", "fp", "fn",
            "unmatched_gt_window_observations",
            "unmatched_predictions_by_tolerance",
            *WINDOW_KEYS
        }:
            raise ValueError("M43 diagnostic fields mismatch in " + condition)
        gt = _nonnegative(d["gt_events"], "GT")
        tp = _nonnegative(d["tp"], "TP")
        fp = _nonnegative(d["fp"], "FP")
        fn = _nonnegative(d["fn"], "FN")
        metrics = m41[condition]["low_to_high"]
        if (gt != ground_truth or tp != metrics["event_tp"] or
            fp != metrics["event_fp"] or fn != metrics["event_fn"] or
            tp + fn != gt):
            raise ValueError("M43 event diagnostics disagree with frozen M41")
        fn_counts = _counts(d["unmatched_gt_window_observations"],
                            FN_KEYS, condition + " FN")
        fp_counts = _counts(d["unmatched_predictions_by_tolerance"],
                            FP_KEYS, condition + " FP")
        if sum(fn_counts.values()) != fn or sum(fp_counts.values()) != fp:
            raise ValueError("M43 diagnostic partition does not match FP/FN")
        values = {name: _nonnegative(d[name], condition + " " + name)
                  for name in WINDOW_KEYS}
        if (any(values[key] > gt for key in WINDOW_KEYS[:5]) or
            values["all_gt_windows_no_usable"] > values["all_gt_windows_usable_lt4"] or
            values["all_gt_windows_usable_lt4"] > values["all_gt_windows_usable_lt6"] or
            values["all_gt_windows_with_low_then_high"] >
            values["all_gt_windows_with_any_low"]):
            raise ValueError("M43 annotated GT window totals inconsistent")
        conditions[condition] = tp
    paired = _counts(data["paired_gt_outcomes"], PAIRED_KEYS, "paired GT")
    if (sum(paired.values()) != ground_truth or
        paired["both_matched"] + paired["dense_only_matched"] !=
        conditions["dense_200ms"] or
        paired["both_matched"] + paired["subsample_only_matched"] !=
        conditions["subsample_400ms"]):
        raise ValueError("M43 paired GT matches disagree with condition TP")


def aggregate_diagnostics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("M43 cannot aggregate empty diagnostics")
    results = {}
    for condition in CONDITIONS:
        entry = {
            "gt_events": 0, "event_tp": 0, "event_fp": 0, "event_fn": 0,
            "unmatched_gt_window_observations": dict.fromkeys(FN_KEYS, 0),
            "unmatched_predictions_by_tolerance": dict.fromkeys(FP_KEYS, 0),
            **dict.fromkeys(WINDOW_KEYS, 0)
        }
        for row in rows:
            d = row[condition]
            for k, source in (("gt_events", "gt_events"),
                              ("event_tp", "tp"), ("event_fp", "fp"),
                              ("event_fn", "fn")):
                entry[k] += d[source]
            for k in WINDOW_KEYS:
                entry[k] += d[k]
            for k in FN_KEYS:
                entry["unmatched_gt_window_observations"][k] += (
                    d["unmatched_gt_window_observations"][k]
                )
            for k in FP_KEYS:
                entry["unmatched_predictions_by_tolerance"][k] += (
                    d["unmatched_predictions_by_tolerance"][k]
                )
        results[condition] = entry
    paired = {name: sum(row["paired_gt_outcomes"][name] for row in rows)
              for name in PAIRED_KEYS}
    return {"conditions": results, "paired_gt_outcomes": paired}
