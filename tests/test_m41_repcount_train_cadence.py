"""M41 offline fixtures: split isolation, stable sampling, paired aggregation."""
from __future__ import annotations

import csv
import io
import json
import tarfile

import pytest

from scripts.m37_repcount_video_benchmark import extract_pose_trace
from scripts.m41_repcount_train_cadence import run, _validate, PROTOCOL


def _add(tar, filename, contents):
    info = tarfile.TarInfo(filename)
    info.size = len(contents)
    tar.addfile(info, io.BytesIO(contents))


def _archive(tmp_path):
    path = tmp_path / "RepCountA.tar.gz"
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["name", "type", "count", "L1", "L2"])
    writer.writeheader()
    writer.writerow({"name": "squat_a.mp4", "type": "squat",
                     "count": "1", "L1": "0", "L2": "20"})
    writer.writerow({"name": "squat_zero.mp4", "type": "squat",
                     "count": "0", "L1": "", "L2": ""})
    with tarfile.open(path, "w:gz") as tar:
        _add(tar, "LLSP/annotation/train.csv", out.getvalue().encode())
        _add(tar, "LLSP/annotation/valid.csv", out.getvalue().encode())
        for name in ["squat_a.mp4", "squat_zero.mp4"]:
            _add(tar, f"LLSP/video/train/{name}", (name * 1000).encode())
            _add(tar, f"LLSP/video/valid/{name}", b"v" * 13000)
    return path


def _method_row(n, pred, tp):
    return {
        "annotated_count": n, "predicted_count": pred,
        "absolute_count_error": abs(pred - n),
        "signed_count_error": pred - n,
        "event_tp": tp, "event_fp": pred - tp, "event_fn": n - tp
    }


def _paired_comparison(trace_path, output_path):
    trace = json.loads(trace_path.read_text())
    n = trace["annotated_count"]
    def condition(samples, predicted):
        return {
            "protocol": "m39-train-only-temporal-return-v1",
            "source_split": "train", "annotation_rep_count": n,
            "input_frame_samples": samples, "pose_sample_coverage": 1.,
            "always_zero": _method_row(n, 0, 0),
            "m34": _method_row(n, 0, 0),
            "low_to_high": _method_row(n, predicted, min(predicted, n)),
        }
    result = {
        "protocol": PROTOCOL, "source_split": "train",
        "annotated_rep_count": n,
        "dense_200ms": condition(20, n),
        "subsample_400ms": condition(10, 0),
        "design": {
            "nominal_dense_ms": 200, "downsample_factor": 2,
            "m34_internal_sample_gate_unchanged_ms": 400,
            "shared_python_pose_tracking_context": True,
            "exact_independent_400ms_pose_estimator_parity": False
        }
    }
    output_path.write_text(json.dumps(result))
    return result


def test_m41_pairwise_comparison_with_two_train_clips_and_zero_negative(tmp_path):
    archive = _archive(tmp_path)
    called = []
    def fake_extractor(path, ann, *, sample_ms):
        assert sample_ms == 200
        called.append((path.read_bytes()[:1], ann["annotated_count"]))
        return {
            "annotated_count": ann["annotated_count"],
            "frames": [{"time_ms": k * 200, "landmarks": None} for k in range(20)]
        }
    report = run(
        archive=archive, root=tmp_path/"private",
        report_path=tmp_path/"report.json", limit=2,
        extract_fn=fake_extractor, compare_fn=_paired_comparison
    )
    assert sorted(count for _,count in called) == [0,1]
    assert all(initial == b"s" for initial,_ in called)
    assert report["source_split"] == "train"
    assert report["videos_evaluated"] == 2
    assert report["annotated_reps"] == 1
    assert report["videos_excluded"] == 0
    assert report["conditions"]["dense_200ms"]["methods"]["low_to_high"]["event_tp"] == 1
    assert report["conditions"]["subsample_400ms"]["methods"]["low_to_high"]["event_fn"] == 1
    assert report["conditions"]["dense_200ms"]["methods"]["low_to_high"]["paired_count_error"] == {
        "denser_better": 1, "tied": 1, "denser_worse": 0
    }
    assert report["conditions"]["dense_200ms"]["methods"]["m34"]["paired_count_error"]["tied"] == 2
    assert report["conditions"]["dense_200ms"]["methods"]["always_zero"]["zero_rep_negative_control_videos"] == 1
    assert not list((tmp_path/"private").glob("m41-traces-*"))
    assert not list((tmp_path/"private").rglob("valid/*.mp4"))
    assert "squat_a" not in (tmp_path/"report.json").read_text()
    assert report["no_valid_or_test_access"] is True
    assert report["original_m34_unchanged"] is True
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        run(archive=archive, root=tmp_path/"private",
            report_path=tmp_path/"report.json")


def test_m41_fails_closed_on_inconsistent_denominators_and_cross_split(tmp_path):
    pair = _paired_comparison(
        type("Trace", (), {"read_text": lambda self: json.dumps({"annotated_count": 1})})(),
        tmp_path/"paired.json"
    )
    _validate(pair, 1)
    broken = json.loads(json.dumps(pair))
    broken["subsample_400ms"]["input_frame_samples"] = 9
    with pytest.raises(ValueError, match="sample lengths"):
        _validate(broken, 1)
    broken = json.loads(json.dumps(pair))
    broken["dense_200ms"]["source_split"] = "validation"
    with pytest.raises(ValueError, match="TRAIN"):
        _validate(broken, 1)
    broken = json.loads(json.dumps(pair))
    broken["design"]["m34_internal_sample_gate_unchanged_ms"] = 200
    with pytest.raises(ValueError, match="design"):
        _validate(broken, 1)


def test_m41_denies_changes_to_prior_400ms_extractor_default():
    with pytest.raises(ValueError, match="200ms or 400ms"):
        extract_pose_trace(None, {}, sample_ms=300)
    with pytest.raises(ValueError, match="200ms or 400ms"):
        extract_pose_trace(None, {}, sample_ms=True)


def test_m41_refuses_zero_or_excessive_clip_limit(tmp_path):
    source = _archive(tmp_path)
    with pytest.raises(ValueError, match="1..20"):
        run(archive=source, root=tmp_path/"private",
            report_path=tmp_path/"r.json", limit=0)
    with pytest.raises(ValueError, match="1..20"):
        run(archive=source, root=tmp_path/"private",
            report_path=tmp_path/"r.json", limit=21)
