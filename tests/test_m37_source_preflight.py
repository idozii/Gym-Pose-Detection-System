"""M37 HF source 404 diagnostics and authorized local dataset fallback tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import m37_repcount_video_benchmark as m37


class VisibleApi:
    def dataset_info(self, repo, revision):
        assert repo == m37.REPO
        return type("Info", (), {"sha": "verifiedsha"})()

    def list_repo_files(self, repo, *, repo_type, revision):
        assert repo_type == "dataset"
        return [
            "README.md", "validation_annotations.json",
            "validation/clip_a.mp4", "train_annotations.json"
        ]


def test_m37_metadata_only_probe_accepts_expected_video_and_event_annotations():
    info = m37.probe_huggingface_source(api=VisibleApi())
    assert info["annotation_file"] == "validation_annotations.json"
    assert info["split_video_files_visible"] == 1
    assert info["no_video_download_performed"] is True
    assert info["revision"] == "verifiedsha"


@pytest.mark.parametrize("contents", [
    ["README.md", "validation/clip_a.mp4"],
    ["README.md", "validation_annotations.json"],
])
def test_m37_preflight_rejects_missing_required_annotation_or_video(contents):
    class ChangedLayoutApi(VisibleApi):
        def list_repo_files(self, *args, **kwargs):
            return contents

    with pytest.raises(RuntimeError, match="visible but does not expose"):
        m37.probe_huggingface_source(api=ChangedLayoutApi())


def test_m37_explains_private_deleted_or_gated_mirror_instead_of_raw_404():
    class UnauthorizedApi(VisibleApi):
        def dataset_info(self, *args, **kwargs):
            class Resp:
                status_code = 404
            error = RuntimeError("Repository Not Found")
            error.response = Resp()
            raise error

    with pytest.raises(RuntimeError, match="cannot be accessed now") as failure:
        m37.probe_huggingface_source(api=UnauthorizedApi())
    message = str(failure.value)
    assert "HTTP status 404" in message
    assert "original" not in message.lower() or "RepCount" in message
    assert "No video" not in message or "No videos" in message


def make_local_dataset(tmp_path):
    path = tmp_path / "repcount_source"
    path.mkdir()
    (path / "validation").mkdir()
    entries = [{
        "video": "squat_trial_01.mp4",
        "action_type": "squat",
        "count": 1,
        "cycle_bounds": [0, 20]
    }]
    (path / "validation_annotations.json").write_text(
        json.dumps(entries), encoding="utf-8"
    )
    (path / "validation" / "squat_trial_01.mp4").write_bytes(b"0"*10001)
    return path


def test_m37_local_authorized_source_downloads_nothing(tmp_path):
    source = make_local_dataset(tmp_path)
    entries, checksum = m37.fetch_local_squat_cohort(
        tmp_path / "outputs", source, split="validation", limit=1
    )
    assert len(entries) == 1
    assert entries[0]["video_path"] == source / "validation" / "squat_trial_01.mp4"
    assert entries[0]["annotated_count"] == 1
    assert len(checksum) == 64


def test_m37_local_dataset_guard_against_symlink_escape(tmp_path):
    source = make_local_dataset(tmp_path)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x"*10001)
    linked = source / "validation" / "squat_trial_01.mp4"
    linked.unlink()
    linked.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes local dataset root"):
        m37.fetch_local_squat_cohort(tmp_path, source)


def test_m37_run_reuses_existing_human_labels_with_no_download(tmp_path):
    source = make_local_dataset(tmp_path)
    captured = []

    def fake_extract(path, annotation):
        captured.append(path)
        return {
            "action": "squat", "fps": 10, "total_frames": 30,
            "annotated_count": 1, "cycle_bounds_frames": [0, 20],
            "frames": [{"time_ms": i * 400, "landmarks": None}
                       for i in range(8)]
        }

    def fake_replay(trace, output):
        assert trace.exists()
        return {
            "annotated_rep_count": 1, "website_observed_cycles": 0,
            "absolute_count_error": 1, "event_tp": 0, "event_fp": 0,
            "event_fn": 1, "pose_sample_coverage": 0
        }

    root = tmp_path / "work"
    report_file = tmp_path / "report.json"
    report = m37.run(
        split="validation", limit=1, root=root, report_path=report_file,
        local_source=source, trace_fn=fake_extract, replay_fn=fake_replay
    )
    assert report["squat_evaluated_videos"] == 1
    assert report["event_recall"] == 0
    assert captured == [source / "validation" / "squat_trial_01.mp4"]
    assert not list(root.glob("m37-traces-*"))
    assert report_file.exists()
    assert "squat_trial_01" not in report_file.read_text(encoding="utf-8")
