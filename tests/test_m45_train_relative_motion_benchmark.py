"""M45 synthetic TRAIN replay tests, with no real publisher videos."""
from __future__ import annotations

import copy
import csv
import io
import json
import tarfile

import pytest

from scripts.m45_train_relative_motion_benchmark import (
    run, validate_row, EXPECTED_RULE, PROTOCOL
)
from scripts.m44_boundary_angle_schema import ANGLE_BIN_KEYS, CONTEXT_KEYS, FP_LOCATION_KEYS


def _add(tf, path, data):
    info = tarfile.TarInfo(path)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def source_archive(tmp_path):
    source = tmp_path / "RepCountA.tar.gz"
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=["name", "type", "count", "L1", "L2"])
    writer.writeheader()
    writer.writerow({
        "name": "source_squat.mp4", "type": "squat",
        "count": 1, "L1": 0, "L2": 10
    })
    csv_bytes = text.getvalue().encode("utf8")
    with tarfile.open(source, "w:gz") as tf:
        _add(tf, "LLSP/annotation/train.csv", csv_bytes)
        _add(tf, "LLSP/annotation/valid.csv", csv_bytes)
        _add(tf, "LLSP/video/valid/source_squat.mp4", b"v" * 14000)
        _add(tf, "LLSP/video/test/source_squat.mp4", b"t" * 14000)
        _add(tf, "LLSP/video/train/source_squat.mp4", b"r" * 14000)
    return source


def row(n, pred, tp):
    precision = tp / pred if pred else None
    recall = tp / n if n else None
    f1 = None if precision is None or recall is None else (
        0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    )
    return {
        "annotated_count": n,
        "predicted_count": pred,
        "absolute_count_error": abs(pred - n),
        "signed_count_error": pred - n,
        "event_tp": tp, "event_fp": pred - tp, "event_fn": n - tp,
        "event_precision": precision, "event_recall": recall, "event_f1": f1
    }


def fake_node(input_path, output_path):
    source = json.loads(input_path.read_text())
    n = source["annotated_count"]
    assert n == 1
    baseline = row(n, 0, 0)
    def original_condition(samples):
        return {
            "protocol": "m39-train-only-temporal-return-v1",
            "source_split": "train", "annotation_rep_count": n,
            "input_frame_samples": samples, "pose_sample_coverage": 1.,
            "m34": baseline, "always_zero": baseline,
            "low_to_high": baseline
        }
    original_m41 = {
        "protocol": "m41-train-cadence-sensitivity-v1",
        "source_split": "train", "annotated_rep_count": n,
        "dense_200ms": original_condition(20),
        "subsample_400ms": original_condition(10),
        "design": {
            "nominal_dense_ms": 200, "downsample_factor": 2,
            "shared_python_pose_tracking_context": True,
            "m34_internal_sample_gate_unchanged_ms": 400,
            "exact_independent_400ms_pose_estimator_parity": False
        }
    }
    def diag():
        return {
            "gt_events": 1, "tp": 0, "fp": 0, "fn": 1,
            "unmatched_gt_window_observations": {
                "no_usable_angle_in_gt_window": 0,
                "usable_but_no_low_in_gt_window": 1,
                "low_but_no_later_high_in_gt_window": 0,
                "low_then_high_in_gt_window_still_unmatched": 0
            },
            "unmatched_predictions_by_tolerance": {
                "unmatched_within_some_gt_end_tolerance": 0,
                "unmatched_outside_all_gt_end_tolerances": 0
            },
            "all_gt_windows_usable_lt4": 1,
            "all_gt_windows_usable_lt6": 1,
            "all_gt_windows_no_usable": 0,
            "all_gt_windows_with_any_low": 0,
            "all_gt_windows_with_low_then_high": 0,
            "side_switch_resets_while_pending": 0,
            "tracking_gap_resets_while_pending": 0
        }
    original_m43 = {
        "protocol": "m43-train-temporal-error-audit-v1",
        "source_split": "train", "annotated_rep_count": n,
        "m41": original_m41,
        "diagnostics": {
            "dense_200ms": diag(), "subsample_400ms": diag(),
            "paired_gt_outcomes": {
                "both_matched": 0, "dense_only_matched": 0,
                "subsample_only_matched": 0, "neither_matched": 1
            }
        }
    }
    def angles():
        return {
            "unmatched_gt_no_low_inside": 1,
            "min_angle_bins_for_no_low_gt": {
                key: 1 if key == "min_gt_angle_gt135_to150" else 0
                for key in ANGLE_BIN_KEYS
            },
            "low_in_750ms_context_for_no_low_gt": {
                key: 1 if key == "low_neither_side" else 0
                for key in CONTEXT_KEYS
            },
            "no_low_gt_with_usable_angle_before": 1,
            "no_low_gt_with_usable_angle_after": 1,
            "unmatched_prediction_location": {key: 0 for key in FP_LOCATION_KEYS}
        }
    original_m44 = {
        "protocol": "m44-train-boundary-angle-observability-v1",
        "source_split": "train", "annotated_rep_count": n,
        "context_ms": 750, "m43": original_m43,
        "conditions": {"dense_200ms": angles(), "subsample_400ms": angles()}
    }
    def candidate_condition(dense):
        gt = {
            "both_matched": 0, "new_only_matched": int(dense),
            "original_only_matched": 0, "neither_matched": int(not dense)
        }
        return {
            "relative_excursion": row(n, int(dense), int(dense)),
            "paired_gt_vs_original_low_to_high": gt,
            "candidate_fp_location": {key: 0 for key in FP_LOCATION_KEYS},
            "pending_resets": {"side": 0, "gap": 0, "timeout": 0}
        }
    result = {
        "protocol": PROTOCOL, "source_split": "train",
        "annotated_rep_count": n, "rule": EXPECTED_RULE,
        "original_m44": original_m44,
        "conditions": {
            "dense_200ms": candidate_condition(True),
            "subsample_400ms": candidate_condition(False)
        }
    }
    output_path.write_text(json.dumps(result), encoding="utf8")
    return result


def test_m45_train_only_candidate_counts_and_privacy(tmp_path):
    selected = []
    def extractor(video, annotation, *, sample_ms):
        selected.append((video.read_bytes()[:1], annotation["annotated_count"], sample_ms))
        return {
            "annotated_count": annotation["annotated_count"],
            "frames": [{"time_ms": i * 200, "landmarks": None} for i in range(20)]
        }
    archive = source_archive(tmp_path)
    out = run(
        archive=archive, root=tmp_path/"private",
        report_path=tmp_path/"report.json", limit=1,
        extract_fn=extractor, compare_fn=fake_node
    )
    assert selected == [(b"r", 1, 200)]
    assert out["source_split"] == "train"
    assert out["videos_evaluated"] == 1
    assert out["videos_excluded"] == 0
    assert out["annotated_reps"] == 1
    dense = out["conditions"]["dense_200ms"]
    assert dense["methods"]["relative_excursion"]["event_tp"] == 1
    assert dense["methods"]["low_to_high"]["event_tp"] == 0
    assert dense["methods"]["relative_excursion"]["event_f1"] == 1.0
    assert dense["paired_diagnostics"]["paired_gt_vs_original_low_to_high"]["new_only_matched"] == 1
    assert dense["paired_diagnostics"]["paired_video_absolute_count_error"] == {
        "relative_better": 1, "tied": 0, "relative_worse": 0
    }
    assert out["no_validation_or_test_access"] is True
    assert out["deployed_rep_counter_unchanged"] is True
    report_text = (tmp_path/"report.json").read_text()
    for term in ('source_squat.mp4', '"frames"', '"time_ms"',
                 '"landmarks"', '"cycle_bounds_frames"', '"video_path"'):
        assert term not in report_text
    assert not list((tmp_path/"private").rglob("valid/*.mp4"))
    assert not list((tmp_path/"private").rglob("test/*.mp4"))
    assert not list((tmp_path/"private").glob("m45-private-*"))
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        run(archive=archive, root=tmp_path/"private",
            report_path=tmp_path/"report.json", limit=1)


def test_m45_fails_closed_on_false_gt_and_modified_rule(tmp_path):
    p = tmp_path/"trace.json"
    p.write_text('{"annotated_count":1}')
    record = fake_node(p, tmp_path/"row.json")
    validate_row(record, 1)
    bad = copy.deepcopy(record)
    bad["rule"]["minDropDeg"] = 20
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_row(bad, 1)
    bad = copy.deepcopy(record)
    bad["conditions"]["dense_200ms"]["relative_excursion"]["event_tp"] = 0
    with pytest.raises(ValueError, match="matched/count totals"):
        validate_row(bad, 1)
    bad = copy.deepcopy(record)
    bad["conditions"]["dense_200ms"]["paired_gt_vs_original_low_to_high"]["new_only_matched"] = 0
    with pytest.raises(ValueError, match="paired matching"):
        validate_row(bad, 1)
    bad = copy.deepcopy(record)
    bad["conditions"]["dense_200ms"]["relative_excursion"]["event_f1"] = 0.3
    with pytest.raises(ValueError, match="event_f1"):
        validate_row(bad, 1)
    bad = copy.deepcopy(record)
    bad["source_split"] = "validation"
    with pytest.raises(ValueError, match="TRAIN GT"):
        validate_row(bad, 1)


def test_m45_rejects_invalid_limit_and_missing_authorized_source(tmp_path):
    with pytest.raises(ValueError, match="1..20"):
        run(archive=tmp_path/"missing.tar.gz", root=tmp_path/"private",
            report_path=tmp_path/"no.json", limit=21)
    with pytest.raises(ValueError, match="owner-local"):
        run(archive=tmp_path/"missing.tar.gz", root=tmp_path/"private",
            report_path=tmp_path/"no.json", limit=5)
