"""M42 offline tests use tiny invented TAR bytes, never genuine clips."""
from __future__ import annotations

import csv
import io
import json
import tarfile

import pytest

from scripts.m37_repcount_tar import selected_from_tar_archive
from scripts.m41_repcount_train_cadence import run, PROTOCOL


def _member(tf, path, data):
    info = tarfile.TarInfo(path)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _fixture(tmp_path, *, with_zero=True):
    path = tmp_path / "RepCountA.tar.gz"
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["name", "type", "count", "L1", "L2"])
    writer.writeheader()
    writer.writerow({
        "name": "squat_positive.mp4", "type": "squat", "count": 1,
        "L1": 0, "L2": 20
    })
    if with_zero:
        writer.writerow({
            "name": "squat_no_reps.mp4", "type": "squat", "count": 0,
            "L1": "", "L2": ""
        })
    data = stream.getvalue().encode("utf8")
    with tarfile.open(path, "w:gz") as tf:
        _member(tf, "LLSP/annotation/train.csv", data)
        _member(tf, "LLSP/annotation/valid.csv", data)
        _member(tf, "LLSP/video/train/squat_positive.mp4", b"p" * 13000)
        _member(tf, "LLSP/video/valid/squat_positive.mp4", b"v" * 13000)
        if with_zero:
            _member(tf, "LLSP/video/valid/squat_no_reps.mp4", b"v" * 13000)
            _member(tf, "LLSP/video/test/squat_no_reps.mp4", b"t" * 13000)
            _member(tf, "LLSP/video/train/squat_no_reps.mp4", b"z" * 13000)
    return path


def _compare_false_dense(trace_path, out_path):
    trace = json.loads(trace_path.read_text())
    assert trace["annotated_count"] == 0
    def method(pred):
        return {
            "annotated_count": 0,
            "predicted_count": pred,
            "absolute_count_error": pred,
            "signed_count_error": pred,
            "event_tp": 0, "event_fp": pred, "event_fn": 0
        }
    def condition(n, low_pred):
        return {
            "protocol": "m39-train-only-temporal-return-v1",
            "source_split": "train", "annotation_rep_count": 0,
            "input_frame_samples": n, "pose_sample_coverage": 0.75,
            "always_zero": method(0), "m34": method(0),
            "low_to_high": method(low_pred)
        }
    result = {
        "protocol": PROTOCOL, "source_split": "train",
        "annotated_rep_count": 0,
        "dense_200ms": condition(20, 1),
        "subsample_400ms": condition(10, 0),
        "design": {
            "nominal_dense_ms": 200,
            "downsample_factor": 2,
            "m34_internal_sample_gate_unchanged_ms": 400,
            "shared_python_pose_tracking_context": True,
            "exact_independent_400ms_pose_estimator_parity": False
        }
    }
    out_path.write_text(json.dumps(result), encoding="utf8")
    return result


def test_m42_selects_only_zero_annotated_train_and_reports_fp(tmp_path):
    archive = _fixture(tmp_path)
    consumed = []

    def extract(video, annotation, *, sample_ms):
        assert sample_ms == 200
        consumed.append((video.read_bytes()[:1], annotation["annotated_count"]))
        return {
            "annotated_count": 0,
            "frames": [{"time_ms": i * 200, "landmarks": None} for i in range(20)]
        }

    result = run(
        archive=archive, root=tmp_path/"private",
        report_path=tmp_path/"m42.json", limit=20, cohort_filter="zero",
        extract_fn=extract, compare_fn=_compare_false_dense
    )

    assert consumed == [(b"z", 0)]
    assert result["source_split"] == "train"
    assert result["source_selection"]["cohort_filter"] == "zero"
    assert result["videos_evaluated"] == 1
    assert result["annotated_reps"] == 0
    assert result["no_valid_or_test_access"] is True

    dense = result["conditions"]["dense_200ms"]["methods"]["low_to_high"]
    sparse = result["conditions"]["subsample_400ms"]["methods"]["low_to_high"]
    assert dense["event_tp"] == 0 and dense["event_fp"] == 1
    assert dense["zero_rep_negative_control_videos"] == 1
    assert dense["zero_rep_negative_controls_with_false_cycles"] == 1
    assert dense["event_precision"] == 0
    assert dense["event_recall"] is None
    assert dense["event_f1"] is None
    assert dense["mean_absolute_count_error"] == 1
    assert dense["paired_count_error"] == {
        "denser_better": 0, "tied": 0, "denser_worse": 1
    }
    assert sparse["event_fp"] == 0
    assert sparse["zero_rep_negative_controls_with_false_cycles"] == 0

    report_text = (tmp_path/"m42.json").read_text()
    for private_value in ("squat_no_reps", "video_path", "landmarks", "frames"):
        assert private_value not in report_text
    assert not list((tmp_path/"private").rglob("valid/*.mp4"))
    assert not list((tmp_path/"private").rglob("test/*.mp4"))
    assert not list((tmp_path/"private").glob("m41-traces-*"))


def test_m42_missing_train_negative_labels_refuses_measurement(tmp_path):
    archive = _fixture(tmp_path, with_zero=False)
    with pytest.raises(ValueError, match="No publisher-annotated zero-rep"):
        run(
            archive=archive, root=tmp_path/"empty",
            report_path=tmp_path/"never.json", cohort_filter="zero", limit=1
        )
    assert not (tmp_path/"never.json").exists()


def test_m42_invalid_count_filter_does_not_select_unsupported_splits(tmp_path):
    archive = _fixture(tmp_path)
    with pytest.raises(ValueError, match="requires an explicit TRAIN split"):
        selected_from_tar_archive(
            archive, tmp_path/"bad", split="validation", limit=1,
            require_explicit_split=True, selection_policy="stable_hash",
            count_filter="zero"
        )
    with pytest.raises(ValueError, match="Unsupported publisher rep-count filter"):
        selected_from_tar_archive(
            archive, tmp_path/"bad", split="train", limit=1,
            require_explicit_split=True, selection_policy="stable_hash",
            count_filter="negative_not_present"
        )
    with pytest.raises(ValueError, match="cohort filter all or zero"):
        run(
            archive=archive, root=tmp_path/"bad",
            report_path=tmp_path/"bad.json",
            cohort_filter="validation"
        )


def test_m42_stable_hash_zero_selection_independent_of_positive_train_order(tmp_path):
    # The existing loader already tests TAR member permutation invariance;
    # verify zero label selection retains the annotation SHA256 and zero count.
    archive = _fixture(tmp_path)
    selected, source_hash = selected_from_tar_archive(
        archive, tmp_path/"only_zero", split="train",
        selection_policy="stable_hash", selection_seed=2026,
        count_filter="zero", require_explicit_split=True, limit=20
    )
    assert len(source_hash) == 64
    assert len(selected) == 1
    assert selected[0]["annotated_count"] == 0
    assert selected[0]["cycle_bounds_frames"] == []
    assert selected[0]["video_path"].read_bytes() == b"z" * 13000
