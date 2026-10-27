"""M44 strict privacy-preserving cohort aggregation for TRAIN angular diagnostics."""
from __future__ import annotations

from scripts.m41_repcount_train_cadence import CONDITIONS

PROTOCOL = "m44-train-boundary-angle-observability-v1"
CONTEXT_MS = 750
ANGLE_BIN_KEYS = (
    "min_gt_angle_gt120_to135", "min_gt_angle_gt135_to150",
    "min_gt_angle_gt150_to165", "min_gt_angle_above165"
)
CONTEXT_KEYS = (
    "low_before_only", "low_after_only",
    "low_both_sides", "low_neither_side"
)
FP_LOCATION_KEYS = (
    "unmatched_inside_any_gt_interval",
    "unmatched_outside_all_gt_intervals"
)
SCALAR_KEYS = (
    "unmatched_gt_no_low_inside",
    "no_low_gt_with_usable_angle_before",
    "no_low_gt_with_usable_angle_after"
)
BLOCK_KEYS = {
    "min_angle_bins_for_no_low_gt": ANGLE_BIN_KEYS,
    "low_in_750ms_context_for_no_low_gt": CONTEXT_KEYS,
    "unmatched_prediction_location": FP_LOCATION_KEYS
}


def _int(x, name):
    if type(x) is not int or x < 0:
        raise ValueError("Invalid nonnegative M44 diagnostic " + name)
    return x


def validate_m44(record: dict) -> None:
    if not isinstance(record, dict) or set(record) != {
        "dense_200ms", "subsample_400ms"
    }:
        raise ValueError("M44 requires exactly two cadence conditions")
    for cond in CONDITIONS:
        obj = record[cond]
        if not isinstance(obj, dict) or set(obj) != set(SCALAR_KEYS) | set(BLOCK_KEYS):
            raise ValueError("M44 diagnostic fields mismatch " + cond)
        n = _int(obj["unmatched_gt_no_low_inside"], "no-low window count")
        for name, keys in BLOCK_KEYS.items():
            value = obj[name]
            if not isinstance(value, dict) or set(value) != set(keys):
                raise ValueError("Invalid M44 diagnostic bins " + name)
            for key in keys:
                _int(value[key], name + "." + key)
            if name != "unmatched_prediction_location" and sum(value.values()) != n:
                raise ValueError("M44 no-low angle/context partition does not reconcile")
        for key in SCALAR_KEYS[1:]:
            if _int(obj[key], key) > n:
                raise ValueError("M44 context coverage exceeds no-low GT count")


def validate_against_m43(record: dict, m43: dict) -> None:
    validate_m44(record)
    if not isinstance(m43, dict):
        raise ValueError("Missing M43 paired diagnostics")
    for cond in CONDITIONS:
        row = record[cond]
        # Per-video M43 diagnostics use cadence keys at their root.
        # Only cohort-aggregated M43 uses a nested "conditions" key.
        prior = m43[cond]
        if row["unmatched_gt_no_low_inside"] != prior[
            "unmatched_gt_window_observations"
        ]["usable_but_no_low_in_gt_window"]:
            raise ValueError("M44 no-low count disagrees with frozen M43")
        if sum(row["unmatched_prediction_location"].values()) != prior["event_fp"]:
            raise ValueError("M44 FP timing count disagrees with frozen M43")


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("M44 requires nonempty source cohort")
    validate_m44(rows[0])
    out = {}
    for cond in CONDITIONS:
        accum = {
            **{k: 0 for k in SCALAR_KEYS},
            **{name: {k: 0 for k in keys} for name, keys in BLOCK_KEYS.items()}
        }
        for row in rows:
            validate_m44(row)
            for key in SCALAR_KEYS:
                accum[key] += row[cond][key]
            for name, keys in BLOCK_KEYS.items():
                for key in keys:
                    accum[name][key] += row[cond][name][key]
        out[cond] = accum
    validate_m44(out)
    return out
