"""M37 original-author ZIP adapter synthetic tests, no real source media."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

import pytest

from scripts.m37_repcount_archive import (
    inspect_archive, parse_official_csv, selected_from_zip
)
from scripts import m37_repcount_video_benchmark as m37


def label_bytes(entries=None):
    if entries is None:
        entries = [
            {"name": "a_squat.mp4", "type": "squat", "count": 2,
             "L1": 0, "L2": 18, "L3": 18, "L4": 40},
            {"name": "b_squat.mp4", "type": "squant", "count": 1,
             "L1": 0, "L2": 22},
            {"name": "c_press.mp4", "type": "front_raise", "count": 1,
             "L1": 0, "L2": 20},
        ]
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=[
        "Unnamed: 0", "type", "name", "count", "L1", "L2", "L3", "L4"
    ])
    writer.writeheader()
    for entry in entries:
        writer.writerow(entry)
    return handle.getvalue().encode("utf-8")


def make_zip(tmp_path, *, labels=True):
    path = tmp_path / "RepCountA.zip"
    with zipfile.ZipFile(path, "w") as zf:
        if labels:
            zf.writestr("RepCountA/new_valid.csv", label_bytes())
        for name in ["a_squat.mp4", "b_squat.mp4", "c_press.mp4"]:
            zf.writestr("RepCountA/valid/" + name, b"x" * 10001)
    return path


def test_m37_original_csv_parser_handles_publisher_float_frame_bounds_and_squant():
    parsed = parse_official_csv(label_bytes())
    assert [p["video"] for p in parsed] == ["a_squat.mp4", "b_squat.mp4"]
    assert parsed[0]["cycle_bounds_frames"] == [0, 18, 18, 40]
    assert parsed[0]["annotated_count"] == 2
    assert parsed[1]["annotated_count"] == 1
    rows = [{
        "name": "bad.mp4", "type": "squat", "count": 2,
        "L1": 0, "L2": 20, "L3": 19, "L4": 30
    }]
    with pytest.raises(ValueError, match="overlap"):
        parse_official_csv(label_bytes(rows))
    rows[0]["L3"] = "19.5"
    with pytest.raises(ValueError, match="integer frames"):
        parse_official_csv(label_bytes(rows))


def test_m37_archive_inspect_metadata_only_and_select_only_squat_videos(tmp_path):
    path = make_zip(tmp_path)
    metadata = inspect_archive(path)
    assert metadata["video_members"] == 3
    assert metadata["csv_members"] == 1
    assert metadata["video_bytes_extracted"] is False
    assert metadata["annotation_csv_members"] == ["RepCountA/new_valid.csv"]
    root = tmp_path / "private_out"
    selected, checksum = selected_from_zip(path, root, limit=2)
    assert len(selected) == 2
    assert len(checksum) == 64
    assert selected[0]["video"] == "a_squat.mp4"
    assert selected[1]["video"] == "b_squat.mp4"
    assert all(x["video_path"].is_file() for x in selected)
    assert not (root / "official_zip_selected" / "validation" / "c_press.mp4").exists()
    assert not (root / "RepCountA").exists()
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        selected_from_zip(path, root, limit=2)


def test_m37_archive_accepts_author_labels_csv_outside_zip(tmp_path):
    path = make_zip(tmp_path, labels=False)
    external = tmp_path / "publisher_new_valid.csv"
    external.write_bytes(label_bytes())
    with pytest.raises(ValueError, match="--annotations-csv"):
        selected_from_zip(path, tmp_path / "nowhere")
    selected, checksum = selected_from_zip(
        path, tmp_path / "private", annotations_csv=external, limit=1
    )
    assert len(selected) == 1
    assert selected[0]["annotated_count"] == 2


def test_m37_archive_refuses_traversal_symlink_duplicate_or_oversized(tmp_path):
    attack = tmp_path / "bad.zip"
    with zipfile.ZipFile(attack, "w") as zf:
        zf.writestr("../escape.mp4", b"some")
    with pytest.raises(ValueError, match="traversal"):
        inspect_archive(attack)
    with zipfile.ZipFile(attack, "w") as zf:
        zf.writestr("valid.csv", label_bytes())
        zf.writestr("val.csv", label_bytes())
        zf.writestr("valid/a_squat.mp4", b"x"*10001)
    with pytest.raises(ValueError, match="Ambiguous"):
        selected_from_zip(attack, tmp_path / "out")
    with zipfile.ZipFile(attack, "w") as zf:
        zf.writestr("new_valid.csv", label_bytes())
        zf.writestr("valid/a_squat.mp4", b"x"*10001)
        zf.writestr("validation/a_squat.mp4", b"x"*10001)
    with pytest.raises(ValueError, match="Duplicate video name"):
        selected_from_zip(attack, tmp_path / "out")
    with zipfile.ZipFile(attack, "w") as zf:
        zf.writestr("new_valid.csv", label_bytes())
        info = zipfile.ZipInfo("valid/a_squat.mp4")
        info.external_attr = 0o120777 << 16
        zf.writestr(info, b"x"*10001)
    with pytest.raises(ValueError, match="symlink"):
        inspect_archive(attack)


def test_m37_zip_source_integrates_without_remote_downloads(tmp_path):
    path = make_zip(tmp_path)
    captured = []

    def fake_extract(video, annotation):
        captured.append(annotation["video"])
        return {
            "action": "squat", "fps": 10, "total_frames": 50,
            "annotated_count": annotation["annotated_count"],
            "cycle_bounds_frames": annotation["cycle_bounds_frames"],
            "frames": [
                {"time_ms": x * 400, "landmarks": None} for x in range(10)
            ]
        }

    def fake_replay(trace, output):
        assert trace.is_file()
        annotation = json.loads(trace.read_text())
        count = annotation["annotated_count"]
        return {
            "annotated_rep_count": count,
            "website_observed_cycles": 0,
            "absolute_count_error": count,
            "event_tp": 0, "event_fp": 0, "event_fn": count,
            "pose_sample_coverage": 0.
        }

    output = m37.run(
        split="validation", limit=1,
        root=tmp_path / "work",
        report_path=tmp_path / "agg.json",
        source_zip=path,
        trace_fn=fake_extract,
        replay_fn=fake_replay
    )
    assert output["source_mode"] == "author_archive"
    assert "user-provided" in output["source"]
    assert output["squat_evaluated_videos"] == 1
    assert output["event_recall"] == 0.
    assert output["third_party_mirror_provenance_independently_verified"] is False
    assert captured == ["a_squat.mp4"]
    report_text = (tmp_path / "agg.json").read_text()
    assert "a_squat.mp4" not in report_text
    assert not list((tmp_path / "work").glob("m37-traces-*"))
