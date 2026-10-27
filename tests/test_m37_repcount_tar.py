"""M37 TAR.GZ regression: synthetic publisher annotation and selected video fixtures.

This test uses tiny artificial bytes; it never downloads or evaluates
real RepCount video. The original compressed file stays on the user's disk.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts import m37_repcount_video_benchmark as m37
from scripts.m37_repcount_tar import (
    inspect_tar_archive, selected_from_tar_archive
)


def annotation():
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=[
        "name", "type", "count", "L1", "L2", "L3", "L4"
    ])
    writer.writeheader()
    writer.writerow({
        "name": "test_squat.mp4", "type": "squat",
        "count": 2, "L1": 0, "L2": 20,
        "L3": 20, "L4": 39
    })
    writer.writerow({
        "name": "squant_alt.mp4", "type": "squant",
        "count": 1, "L1": 0, "L2": 27
    })
    writer.writerow({
        "name": "test_press.mp4", "type": "front_raise",
        "count": 1, "L1": 0, "L2": 29
    })
    return text.getvalue().encode("utf-8")


def _add_bytes(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def sample_tar(tmp_path, with_csv=True, duplicate=False):
    tar_path = tmp_path / "RepCountA.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        if with_csv:
            _add_bytes(tf, "RepCount-A/new_valid.csv", annotation())
        _add_bytes(tf, "RepCount-A/valid/test_press.mp4", b"z" * 11000)
        _add_bytes(tf, "RepCount-A/valid/test_squat.mp4", b"v" * 11000)
        _add_bytes(tf, "RepCount-A/valid/squant_alt.mp4", b"w" * 11000)
        if duplicate:
            _add_bytes(tf, "RepCount-A/valid/duplicate/test_squat.mp4", b"v" * 11000)
    return tar_path


def test_inspection_handles_real_tar_gz_extension_and_no_extraction(tmp_path):
    archive = sample_tar(tmp_path)
    summary = inspect_tar_archive(archive)
    assert summary["archive_format"] == "tar.gz"
    assert summary["video_members"] == 3
    assert summary["csv_members_visible_first_25"] == ["RepCount-A/new_valid.csv"]
    assert summary["video_bytes_extracted"] is False
    assert summary["full_tar_gz_stream_scanned"] is True
    assert not (tmp_path / "official_tar_selected").exists()


def test_embedded_original_labels_select_only_preannotated_squat(tmp_path):
    archive = sample_tar(tmp_path)
    root = tmp_path / "local_private"
    selected, digest = selected_from_tar_archive(archive, root, limit=2)
    assert len(selected) == 2
    assert {row["video"] for row in selected} == {
        "test_squat.mp4", "squant_alt.mp4"
    }
    assert all(row["video_path"].stat().st_size == 11000 for row in selected)
    assert len(digest) == 64
    assert not (root / "official_tar_selected" / "validation" / "test_press.mp4").exists()


def test_publisher_csv_outside_archive_is_one_pass_and_no_human_labels(tmp_path):
    archive = sample_tar(tmp_path, with_csv=False)
    csv_file = tmp_path / "official_new_valid.csv"
    csv_file.write_bytes(annotation())
    selected, _ = selected_from_tar_archive(
        archive, tmp_path / "local", limit=1, annotations_csv=csv_file
    )
    assert selected[0]["video"] == "test_squat.mp4"
    with pytest.raises(ValueError, match="--annotations-csv"):
        selected_from_tar_archive(archive, tmp_path / "missing")


def test_tar_archive_security_rejects_traversal_symlinks_and_oversize(tmp_path):
    path = tmp_path / "attack.tgz"
    with tarfile.open(path, "w:gz") as tf:
        _add_bytes(tf, "../escape.mp4", b"x" * 10001)
    with pytest.raises(ValueError, match="traversal"):
        inspect_tar_archive(path)
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo("RepCount-A/valid/linked.mp4")
        info.type = tarfile.SYMTYPE
        info.linkname = "/tmp/secrets"
        tf.addfile(info)
    with pytest.raises(ValueError, match="symlink"):
        inspect_tar_archive(path)
    with tarfile.open(path, "w:gz") as tf:
        _add_bytes(tf, "RepCount-A/new_valid.csv", annotation())
        big_video = tarfile.TarInfo("RepCount-A/valid/test_squat.mp4")
        big_video.size = 251 * 1024 * 1024
        # Cannot synthesize 251MB in a small fixture; test the actual size
        # gate directly without materializing the archive.
    from scripts.m37_repcount_tar import MAX_SINGLE_VIDEO_BYTES, _copy_selected
    assert big_video.size > MAX_SINGLE_VIDEO_BYTES
    with pytest.raises(ValueError, match="size cap"):
        _copy_selected(None, big_video, tmp_path / "no_output" / "clip.mp4")


def test_duplicate_annotated_video_members_reject_ambiguity(tmp_path):
    archive = sample_tar(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate selected"):
        selected_from_tar_archive(archive, tmp_path / "private", limit=3)


def test_cli_can_inspect_tar_gz_under_original_command_name(tmp_path, capsys):
    archive = sample_tar(tmp_path)
    status = m37.main(["--inspect-source-zip", str(archive)])
    assert status == 0
    output = json.loads(capsys.readouterr().out)
    assert output["archive_format"] == "tar.gz"
    assert output["video_members"] == 3


def test_tar_path_benchmark_reuses_M34_pipeline_without_network(tmp_path):
    archive = sample_tar(tmp_path)
    seen = []
    def extractor(video, item):
        seen.append(item["video"])
        return {
            "action": "squat", "fps": 10, "total_frames": 60,
            "annotated_count": item["annotated_count"],
            "cycle_bounds_frames": item["cycle_bounds_frames"],
            "frames": [
                {"time_ms": j*400, "landmarks": None} for j in range(10)
            ]
        }

    def replayer(trace, output):
        item = json.loads(trace.read_text(encoding="utf-8"))
        count = item["annotated_count"]
        return {
            "annotated_rep_count": count,
            "website_observed_cycles": 0,
            "absolute_count_error": count,
            "event_tp": 0, "event_fp": 0,
            "event_fn": count, "pose_sample_coverage": 0
        }

    result = m37.run(
        split="validation", limit=1,
        root=tmp_path / "work", report_path=tmp_path / "report.json",
        source_zip=archive, trace_fn=extractor, replay_fn=replayer
    )
    assert result["source_mode"] == "author_archive"
    assert result["squat_evaluated_videos"] == 1
    assert result["event_recall"] == 0
    assert len(seen) == 1
    assert not list((tmp_path / "work").glob("m37-traces-*"))
    assert "test_squat.mp4" not in (tmp_path / "report.json").read_text()
