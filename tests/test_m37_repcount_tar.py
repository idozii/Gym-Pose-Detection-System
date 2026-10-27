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



def test_inspected_llsp_layout_uses_embedded_valid_csv_and_correct_video_split(tmp_path):
    """Small synthetic fixture matching the owner's observed LLSP member layout.

    Historical publisher CSVs include an unnamed index column and integer-valued
    decimal frame bounds. Same-named test clips must not leak into validation.
    """
    archive = tmp_path / "RepCountA.tar.gz"
    valid_csv = (
        ",type,name,count,L1,L2,L3,L4\n"
        "0,squat,stu4_70.mp4,2,5,25,25.0,50.0\n"
        "1,squant,train3946.mp4,1,0,30,,\n"
        "2,front_raise,other.mp4,1,1,20,,\n"
    ).encode("utf-8")
    with tarfile.open(archive, "w:gz") as tf:
        _add_bytes(tf, "LLSP/annotation/test.csv", annotation())
        _add_bytes(tf, "LLSP/annotation/train.csv", annotation())
        _add_bytes(tf, "LLSP/annotation/valid.csv", valid_csv)
        _add_bytes(tf, "LLSP/video/test/stu4_70.mp4", b"t" * 11000)
        _add_bytes(tf, "LLSP/video/valid/stu4_70.mp4", b"v" * 11000)
        _add_bytes(tf, "LLSP/video/valid/train3946.mp4", b"w" * 11000)
        _add_bytes(tf, "LLSP/video/valid/other.mp4", b"x" * 11000)
    preview = inspect_tar_archive(archive, split="validation")
    assert preview["video_members"] == 4
    assert "LLSP/annotation/valid.csv" in preview["csv_members_visible_first_25"]
    selected, digest = selected_from_tar_archive(
        archive, tmp_path / "private", split="validation", limit=2
    )
    assert len(digest) == 64
    assert {clip["video"] for clip in selected} == {
        "stu4_70.mp4", "train3946.mp4"
    }
    assert next(c for c in selected if c["video"] == "stu4_70.mp4")[
        "video_path"
    ].read_bytes() == b"v" * 11000
    assert not list((tmp_path / "private").rglob("other.mp4"))


def test_wrong_external_action_csv_is_rejected_not_relabelled(tmp_path):
    archive = sample_tar(tmp_path, with_csv=False)
    external_csv = tmp_path / "not_squat.csv"
    external_csv.write_text(
        "type,name,count,L1,L2\n"
        "front_raise,test_squat.mp4,1,0,20\n",
        encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no labeled Squat"):
        selected_from_tar_archive(
            archive, tmp_path / "private", annotations_csv=external_csv
        )

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



def test_actual_LLSP_style_tar_zero_rep_annotation_is_supported(tmp_path):
    # The public LLSP valid split contains a Squat entry like stu9_67.mp4
    # count=0 with empty L fields: a valid negative-control video.
    csv_stream = io.StringIO()
    writer = csv.DictWriter(csv_stream, fieldnames=[
        "name", "type", "count", "L1", "L2"
    ])
    writer.writeheader()
    writer.writerow({
        "name": "stu9_67.mp4", "type": "squat",
        "count": "0", "L1": "", "L2": ""
    })
    archive = tmp_path / "RepCountA.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        _add_bytes(tf, "LLSP/annotation/valid.csv",
                   csv_stream.getvalue().encode("utf-8"))
        _add_bytes(tf, "LLSP/video/valid/stu9_67.mp4", b"v" * 12000)
    selected, source_hash = selected_from_tar_archive(
        archive, tmp_path / "local_private", limit=1
    )
    assert selected[0]["annotated_count"] == 0
    assert selected[0]["cycle_bounds_frames"] == []
    assert selected[0]["video_path"].is_file()
    assert len(source_hash) == 64



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


def test_m39_explicit_train_video_policy_ignores_splitless_ambiguous_members(tmp_path):
    archive = tmp_path / "RepCountA.tar.gz"
    train = (
        "name,type,count,L1,L2\n"
        "example.mp4,squat,1,0,20\n"
    ).encode("utf-8")
    with tarfile.open(archive, "w:gz") as tf:
        _add_bytes(tf, "LLSP/annotation/train.csv", train)
        _add_bytes(tf, "LLSP/video/example.mp4", b"n" * 12000)
        _add_bytes(tf, "LLSP/video/valid/example.mp4", b"v" * 12000)
    with pytest.raises(ValueError, match="No pre-annotated Squat"):
        selected_from_tar_archive(
            archive, tmp_path/"strict", split="train", limit=1,
            require_explicit_split=True
        )


def test_m40_stable_hash_selection_invariant_to_archive_order_and_split(tmp_path):
    import hashlib
    names = [f"train_clip_{i}.mp4" for i in range(8)]
    publisher_csv = ("name,type,count,L1,L2\n" + "".join(
        f"{name},squat,1,0,20\n" for name in names
    )).encode("utf-8")
    results = []
    for i, order in enumerate((names, list(reversed(names)))):
        archive = tmp_path / f"RepCount{i}.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            _add_bytes(tf, "LLSP/annotation/train.csv", publisher_csv)
            for name in order:
                _add_bytes(tf, f"LLSP/video/train/{name}", (name * 750).encode())
                _add_bytes(tf, f"LLSP/video/valid/{name}", b"v" * 12000)
        selected, source_sha = selected_from_tar_archive(
            archive, tmp_path / f"private_{i}", split="train",
            limit=3, require_explicit_split=True,
            selection_policy="stable_hash", selection_seed=2026
        )
        result_names = {row["video"] for row in selected}
        results.append((result_names, source_sha))
        assert len(result_names) == 3
        assert all(row["video_path"].read_bytes().startswith(b"train_clip")
                   for row in selected)
        assert not list((tmp_path / f"private_{i}").rglob("valid/*.mp4"))
    assert results[0] == results[1]
    expected = set(sorted(names, key=lambda name: (
        hashlib.sha256(f"repcount-m40-train-v1:2026:{name}".encode()).digest(),
        name
    ))[:3])
    assert results[0][0] == expected


def test_m40_stable_hash_requires_explicit_split_and_valid_seed(tmp_path):
    archive = sample_tar(tmp_path)
    with pytest.raises(ValueError, match="explicit source split"):
        selected_from_tar_archive(
            archive, tmp_path / "private", split="validation",
            limit=2, selection_policy="stable_hash"
        )
    with pytest.raises(ValueError, match="Selection seed"):
        selected_from_tar_archive(
            archive, tmp_path / "private", split="train",
            limit=2, require_explicit_split=True,
            selection_policy="stable_hash", selection_seed=-1
        )
