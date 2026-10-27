"""M43 synthetic TAR + aggregate semantic checks. Never uses real RepCount data."""
from __future__ import annotations

import csv
import io
import json
import tarfile

import pytest

from scripts.m43_repcount_train_error_audit import run, PROTOCOL
from scripts.m43_temporal_error_schema import (
    FN_KEYS, FP_KEYS, validate_m43_diagnostics
)


def _add(tf, path, data):
    info = tarfile.TarInfo(path)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def archive(tmp_path):
    path = tmp_path / "RepCountA.tar.gz"
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=[
        "name", "type", "count", "L1", "L2"
    ])
    writer.writeheader()
    writer.writerow({
        "name": "train_squat.mp4", "type": "squat",
        "count": 1, "L1": 0, "L2": 20
    })
    csv_data = csv_buffer.getvalue().encode("utf8")
    with tarfile.open(path, "w:gz") as tf:
        _add(tf, "LLSP/annotation/train.csv", csv_data)
        _add(tf, "LLSP/annotation/valid.csv", csv_data)
        _add(tf, "LLSP/video/valid/train_squat.mp4", b"v" * 12000)
        _add(tf, "LLSP/video/test/train_squat.mp4", b"t" * 12000)
        _add(tf, "LLSP/video/train/train_squat.mp4", b"r" * 12000)
    return path


def method_row(n, pred, tp):
    return {
        "annotated_count": n, "predicted_count": pred,
        "absolute_count_error": abs(n - pred),
        "signed_count_error": pred - n,
        "event_tp": tp, "event_fp": pred - tp, "event_fn": n - tp
    }


def fake_node(input_path, output_path):
    trace = json.loads(input_path.read_text())
    n = trace["annotated_count"]
    def baseline(samples, detected):
        return {
            "protocol": "m39-train-only-temporal-return-v1",
            "source_split": "train", "annotation_rep_count": n,
            "input_frame_samples": samples, "pose_sample_coverage": 0.5,
            "m34": method_row(n, 0, 0),
            "always_zero": method_row(n, 0, 0),
            "low_to_high": method_row(n, detected, detected),
        }
    m41 = {
        "protocol": "m41-train-cadence-sensitivity-v1",
        "source_split": "train", "annotated_rep_count": n,
        "dense_200ms": baseline(20, n),
        "subsample_400ms": baseline(10, 0),
        "design": {
            "nominal_dense_ms": 200, "downsample_factor": 2,
            "shared_python_pose_tracking_context": True,
            "m34_internal_sample_gate_unchanged_ms": 400,
            "exact_independent_400ms_pose_estimator_parity": False
        }
    }
    def diagnostics(tp):
        fn = n - tp
        return {
            "gt_events": n, "tp": tp, "fp": 0, "fn": fn,
            "unmatched_gt_window_observations": {
                k: fn if k == FN_KEYS[0] else 0 for k in FN_KEYS
            },
            "unmatched_predictions_by_tolerance": {k: 0 for k in FP_KEYS},
            "all_gt_windows_usable_lt4": n,
            "all_gt_windows_usable_lt6": n,
            "all_gt_windows_no_usable": n,
            "all_gt_windows_with_any_low": 0,
            "all_gt_windows_with_low_then_high": 0,
            "side_switch_resets_while_pending": 0,
            "tracking_gap_resets_while_pending": 0
        }
    result = {
        "protocol": PROTOCOL, "source_split": "train",
        "annotated_rep_count": n, "m41": m41,
        "diagnostics": {
            "dense_200ms": diagnostics(n),
            "subsample_400ms": diagnostics(0),
            "paired_gt_outcomes": {
                "both_matched": 0, "dense_only_matched": n,
                "subsample_only_matched": 0, "neither_matched": 0
            }
        }
    }
    output_path.write_text(json.dumps(result))
    return result


def test_m43_replays_only_train_and_persists_no_event_timestamps(tmp_path):
    seen = []
    def extractor(path, annotation, *, sample_ms):
        assert sample_ms == 200
        seen.append(path.read_bytes()[:1])
        return {
            "annotated_count": annotation["annotated_count"],
            "frames": [{"time_ms": i * 200, "landmarks": None} for i in range(20)]
        }
    report = run(
        archive=archive(tmp_path), root=tmp_path / "local",
        report_path=tmp_path / "aggregate.json", limit=1,
        extract_fn=extractor, compare_fn=fake_node
    )
    assert seen == [b"r"]
    assert report["source_split"] == "train"
    assert report["videos_evaluated"] == 1
    assert report["videos_excluded"] == 0
    assert report["annotated_reps"] == 1
    assert report["m43_diagnostics"]["conditions"]["dense_200ms"]["event_tp"] == 1
    assert report["m43_diagnostics"]["conditions"]["subsample_400ms"]["event_fn"] == 1
    assert report["m43_diagnostics"]["paired_gt_outcomes"]["dense_only_matched"] == 1
    assert report["frozen_m41_method_metrics"]["dense_200ms"]["methods"]["low_to_high"]["event_tp"] == 1
    assert report["validation_or_test_accessed"] is False
    assert report["original_m34_modified"] is False
    text = (tmp_path/"aggregate.json").read_text()
    assert "train_squat.mp4" not in text
    assert '"landmarks"' not in text
    assert '"frames"' not in text
    assert '"time_ms"' not in text
    assert '"cycle_bounds_frames"' not in text
    assert not list((tmp_path/"local").rglob("valid/*.mp4"))
    assert not list((tmp_path/"local").rglob("test/*.mp4"))
    assert not list((tmp_path/"local").glob("m43-private-*"))
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        run(archive=tmp_path/"RepCountA.tar.gz",
            root=tmp_path/"local", report_path=tmp_path/"aggregate.json")


def test_m43_rejects_inconsistent_missed_rep_partition(tmp_path):
    trace_path = tmp_path/"trace.json"
    trace_path.write_text('{"annotated_count":1}')
    data = fake_node(trace_path, tmp_path/"node.json")
    validate_m43_diagnostics(data["diagnostics"], data["m41"])
    corrupt = json.loads(json.dumps(data))
    corrupt["diagnostics"]["subsample_400ms"][
        "unmatched_gt_window_observations"][FN_KEYS[0]] = 0
    with pytest.raises(ValueError, match="partition"):
        validate_m43_diagnostics(corrupt["diagnostics"], corrupt["m41"])
    corrupt = json.loads(json.dumps(data))
    corrupt["diagnostics"]["paired_gt_outcomes"]["dense_only_matched"] = 0
    with pytest.raises(ValueError, match="paired GT"):
        validate_m43_diagnostics(corrupt["diagnostics"], corrupt["m41"])
    corrupt = json.loads(json.dumps(data))
    corrupt["diagnostics"]["dense_200ms"]["fp"] = 1
    with pytest.raises(ValueError, match="frozen M41"):
        validate_m43_diagnostics(corrupt["diagnostics"], corrupt["m41"])
    corrupt = json.loads(json.dumps(data))
    corrupt["diagnostics"]["dense_200ms"]["all_gt_windows_usable_lt4"] = 2
    with pytest.raises(ValueError, match="window"):
        validate_m43_diagnostics(corrupt["diagnostics"], corrupt["m41"])


def test_m43_fails_on_excluded_dataset_or_invalid_limit(tmp_path):
    with pytest.raises(ValueError, match="1..20"):
        run(archive=tmp_path/"absent.tar.gz", root=tmp_path/"local",
            report_path=tmp_path/"bad.json", limit=21)
    with pytest.raises(ValueError, match="local"):
        run(archive=tmp_path/"absent.tar.gz", root=tmp_path/"local",
            report_path=tmp_path/"bad.json", limit=5)


def test_m43_refuses_fake_node_cross_split_and_missing_per_video_metrics(tmp_path):
    def bad_node(input_path, out_path):
        result = fake_node(input_path, out_path)
        result["source_split"] = "validation"
        return result
    def extractor(path, annotation, *, sample_ms):
        return {
            "annotated_count": annotation["annotated_count"],
            "frames": [{"time_ms": i * 200, "landmarks": None} for i in range(20)]
        }
    with pytest.raises(ValueError, match="No TRAIN videos evaluated"):
        run(
            archive=archive(tmp_path), root=tmp_path/"local",
            report_path=tmp_path/"missing.json", limit=1,
            extract_fn=extractor, compare_fn=bad_node
        )
    assert not (tmp_path/"missing.json").exists()
