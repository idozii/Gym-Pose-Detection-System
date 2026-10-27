"""M39 source-split and report controls: pure synthetic offline fixtures."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts.m39_repcount_train_baselines import run, _validate_paired


def _add(tar, name, contents):
    info = tarfile.TarInfo(name)
    info.size = len(contents)
    tar.addfile(info, io.BytesIO(contents))


def _archive(tmp_path):
    p = tmp_path / "RepCountA.tar.gz"
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=[
        "name", "type", "count", "L1", "L2"
    ])
    writer.writeheader()
    writer.writerow({"name":"train_squat.mp4","type":"squat",
                     "count":1,"L1":0,"L2":18})
    data = csv_buffer.getvalue().encode("utf8")
    with tarfile.open(p, "w:gz") as tar:
        _add(tar, "LLSP/annotation/train.csv", data)
        _add(tar, "LLSP/annotation/valid.csv", data)
        _add(tar, "LLSP/video/train/train_squat.mp4", b"train" * 2600)
        _add(tar, "LLSP/video/valid/train_squat.mp4", b"valid" * 2600)
    return p


def _fake_compare(trace_file, out_file):
    trace = json.loads(trace_file.read_text())
    n = trace["annotated_count"]
    def row(pred, tp):
        return {
            "annotated_count":n,"predicted_count":pred,
            "absolute_count_error":abs(n-pred),
            "signed_count_error":pred-n,
            "event_tp":tp,"event_fp":pred-tp,"event_fn":n-tp
        }
    output = {
        "protocol":"m39-train-only-temporal-return-v1",
        "source_split":"train", "annotation_rep_count":n,
        "pose_sample_coverage":.5,
        "m34":row(0,0), "low_to_high":row(1,1),
        "always_zero":row(0,0)
    }
    out_file.write_text(json.dumps(output))
    return output


def test_m39_selects_only_train_publisher_clips_and_never_public_valid_test(tmp_path):
    archive = _archive(tmp_path)
    captured = []
    def extractor(path, ann):
        captured.append(path.read_bytes()[:5])
        return {"annotated_count":ann["annotated_count"], "frames":[]}
    report = run(
        archive=archive, root=tmp_path / "private",
        report_path=tmp_path / "out.json", limit=1,
        extract_fn=extractor, compare_fn=_fake_compare
    )
    assert captured == [b"train"]
    assert report["source_split"] == "train"
    assert report["candidate_videos"] == 1
    assert report["evaluated_videos"] == 1
    assert report["results"]["low_to_high"]["event_tp"] == 1
    assert report["results"]["m34"]["event_recall"] == 0
    assert report["results"]["always_zero"]["event_precision"] is None
    assert report["validation_or_test_accessed"] is False
    assert report["original_m34_modified"] is False
    content = (tmp_path / "out.json").read_text()
    assert "train_squat" not in content
    assert "video_path" not in content
    assert not list((tmp_path / "private").glob("m39-trace-*"))
    assert not list((tmp_path / "private").rglob("valid/*.mp4"))
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        run(archive=archive, root=tmp_path/"private",
            report_path=tmp_path/"out.json")


def test_m39_refuses_splitless_video_and_missing_train_annotation(tmp_path):
    p = tmp_path / "RepCountA.tar.gz"
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=[
        "name","type","count","L1","L2"
    ])
    writer.writeheader()
    writer.writerow({"name":"clip.mp4","type":"squat",
                     "count":1,"L1":0,"L2":20})
    with tarfile.open(p, "w:gz") as tar:
        _add(tar,"LLSP/annotation/train.csv",
             csv_buffer.getvalue().encode())
        _add(tar,"LLSP/video/clip.mp4",b"x"*12000)
        _add(tar,"LLSP/video/valid/clip.mp4",b"v"*12000)
    with pytest.raises(ValueError, match="No pre-annotated Squat"):
        run(archive=p, root=tmp_path/"private",
            report_path=tmp_path/"missing.json")


def test_m39_cannot_turn_bad_fake_baseline_results_into_real_accuracy():
    good = {
        "protocol":"m39-train-only-temporal-return-v1",
        "source_split":"train","annotation_rep_count":1,
        "pose_sample_coverage": .5,
        "m34":{"annotated_count":1,"predicted_count":0,
               "absolute_count_error":1,"signed_count_error":-1,
               "event_tp":0,"event_fp":0,"event_fn":1},
        "low_to_high":{"annotated_count":1,"predicted_count":1,
                       "absolute_count_error":0,"signed_count_error":0,
                       "event_tp":1,"event_fp":0,"event_fn":0},
        "always_zero":{"annotated_count":1,"predicted_count":0,
                       "absolute_count_error":1,"signed_count_error":-1,
                       "event_tp":0,"event_fp":0,"event_fn":1}
    }
    _validate_paired(good, 1)
    broken = json.loads(json.dumps(good))
    broken["low_to_high"]["event_tp"] = 2
    with pytest.raises(ValueError, match="mismatch"):
        _validate_paired(broken, 1)
    broken = json.loads(json.dumps(good))
    broken["source_split"] = "validation"
    with pytest.raises(ValueError, match="TRAIN"):
        _validate_paired(broken, 1)
