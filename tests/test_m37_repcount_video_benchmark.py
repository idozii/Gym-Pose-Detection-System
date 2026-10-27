"""M37 synthetic offline tests; no real video downloads or personal data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import m37_repcount_video_benchmark as m37


def source_annotations():
    return [
        {
            "video": "squat01.mp4", "video_id": "squat01",
            "action_type": "squat", "count": 2,
            "cycle_bounds": [0, 20, 20, 36]
        },
        {
            "video": "squat02.mp4", "video_id": "squat02",
            "action_type": "squat", "count": 1,
            "cycle_bounds": [0, 25]
        },
        {
            "video": "ohp01.mp4", "action_type": "front_raise",
            "count": 1, "cycle_bounds": [0, 20]
        }
    ]


def test_m37_rejects_missing_or_misaligned_paired_event_annotation(tmp_path):
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps(source_annotations()))
    parsed = m37.read_mirror_annotations(annotation)
    assert len(parsed) == 2
    assert [row["video"] for row in parsed] == ["squat01.mp4", "squat02.mp4"]
    assert parsed[0]["cycle_bounds_frames"] == [0, 20, 20, 36]
    cases = [
        ({"count": 3}, "exactly count"),
        ({"cycle_bounds": [0, 20, 22]}, "exactly count"),
        ({"cycle_bounds": [0, 20, 19, 36]}, "overlap"),
        ({"video": "../secret.mp4"}, "unsafe"),
    ]
    for change, expected in cases:
        wrong = source_annotations()
        wrong[0] = {**wrong[0], **change}
        annotation.write_text(json.dumps(wrong))
        with pytest.raises(ValueError, match=expected):
            m37.read_mirror_annotations(annotation)


def test_m37_only_chooses_small_deterministic_squat_subset(tmp_path):
    requested = []
    def fake_download(*, repo_id, filename, repo_type, revision, local_dir):
        assert repo_id == m37.REPO
        assert repo_type == "dataset"
        requested.append(filename)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith(".json"):
            path.write_text(json.dumps(source_annotations()), encoding="utf-8")
        else:
            path.write_bytes(b"0" * 10001)
        return str(path)
    entries, digest = m37.fetch_small_squat_cohort(
        tmp_path, split="validation", limit=1, download_fn=fake_download
    )
    assert requested == ["validation_annotations.json", "validation/squat01.mp4"]
    assert len(entries) == 1
    assert entries[0]["annotated_count"] == 2
    assert len(digest) == 64
    with pytest.raises(ValueError, match="limit"):
        m37.fetch_small_squat_cohort(tmp_path, limit=21,
                                     download_fn=fake_download)


def test_m37_metrics_aggregate_true_event_counts_and_exclusions():
    clips = [
        {
            "annotated_rep_count": 2, "website_observed_cycles": 3,
            "absolute_count_error": 1, "event_tp": 2, "event_fp": 1,
            "event_fn": 0, "pose_sample_coverage": .75
        },
        {
            "annotated_rep_count": 3, "website_observed_cycles": 1,
            "absolute_count_error": 2, "event_tp": 1, "event_fp": 0,
            "event_fn": 2, "pose_sample_coverage": .5
        }
    ]
    out = m37.summarise(
        clips, {"ValueError": 1}, split="validation",
        revision="main", source_hash="x"*64, candidates=3
    )
    assert out["squat_evaluated_videos"] == 2
    assert out["excluded_videos"] == 1
    assert out["source_total_annotated_reps"] == 5
    assert out["m34_total_observed_cycles"] == 4
    assert out["event_tp"] == 3
    assert out["event_fp"] == 1
    assert out["event_fn"] == 2
    assert out["event_precision"] == .75
    assert out["event_recall"] == .6
    assert out["session_mean_absolute_count_error"] == 1.5
    assert out["session_exact_count_fraction"] == 0
    assert out["session_within_one_fraction"] == .5
    assert out["real_video_measured"] is True
    assert out["browser_python_pose_estimator_parity_validated"] is False
    assert out["third_party_mirror_provenance_independently_verified"] is False
    assert out["original_m20_val_or_test_used"] is False
    with pytest.raises(ValueError, match="No RepCount Squat"):
        m37.summarise([], {"ValueError": 2}, split="validation",
                      revision="main", source_hash="", candidates=2)


def test_m37_full_orchestrator_with_fake_download_extraction_and_replay(tmp_path):
    requests = []
    def fake_download(*, repo_id, filename, repo_type, revision, local_dir):
        requests.append(filename)
        path = Path(local_dir)/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith(".json"):
            path.write_text(json.dumps(source_annotations()))
        else:
            path.write_bytes(b"fake"*3000)
        return str(path)

    extracted = []
    def fake_extract(path, annotation):
        extracted.append(annotation["video"])
        return {
            "fps": 10, "total_frames": 40, "action": "squat",
            "annotated_count": annotation["annotated_count"],
            "cycle_bounds_frames": annotation["cycle_bounds_frames"],
            "frames": [
                {"time_ms": i*400, "landmarks": None} for i in range(10)
            ]
        }

    replayed = []
    def fake_replay(trace_file, result_file):
        assert trace_file.exists()
        trace = json.loads(trace_file.read_text())
        replayed.append(trace["annotated_count"])
        out = {
            "protocol": m37.PROTOCOL,
            "annotated_rep_count": trace["annotated_count"],
            "website_observed_cycles": trace["annotated_count"],
            "absolute_count_error": 0,
            "event_tp": trace["annotated_count"], "event_fp": 0,
            "event_fn": 0, "pose_sample_coverage": .8
        }
        result_file.write_text(json.dumps(out))
        return out

    root = tmp_path/"data"/"external"
    report_path = tmp_path/"data"/"processed"/"report.json"
    output = m37.run(
        split="validation", limit=2, root=root, report_path=report_path,
        download_fn=fake_download, trace_fn=fake_extract,
        replay_fn=fake_replay
    )
    assert output["squat_evaluated_videos"] == 2
    assert output["event_f1"] == 1
    assert len(requests) == 3
    assert extracted == ["squat01.mp4", "squat02.mp4"]
    assert replayed == [2, 1]
    assert report_path.is_file()
    assert not list(root.glob("m37-traces-*"))
    raw = report_path.read_text()
    assert "squat01.mp4" not in raw
    assert "squat02.mp4" not in raw
    assert "landmarks" not in raw
    with pytest.raises(FileExistsError):
        m37.run(
            split="validation", limit=2, root=root, report_path=report_path,
            download_fn=fake_download, trace_fn=fake_extract,
            replay_fn=fake_replay
        )


def test_m37_decoding_failure_is_reported_as_exclusion(tmp_path):
    def fake_download(*, repo_id, filename, repo_type, revision, local_dir):
        path = Path(local_dir)/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith(".json"):
            path.write_text(json.dumps(source_annotations()))
        else:
            path.write_bytes(b"x"*10010)
        return str(path)

    def fail_first(path, annotation):
        if annotation["video"] == "squat01.mp4":
            raise ValueError("Source frame mismatch")
        return {"annotated_count": 1}

    def fake_replay(trace_path, result_path):
        return {
            "annotated_rep_count": 1, "website_observed_cycles": 0,
            "absolute_count_error": 1, "event_tp": 0,
            "event_fp": 0, "event_fn": 1,
            "pose_sample_coverage": 0
        }

    result = m37.run(
        split="validation", limit=2, root=tmp_path/"external",
        report_path=tmp_path/"processed"/"report.json",
        download_fn=fake_download, trace_fn=fail_first,
        replay_fn=fake_replay
    )
    assert result["squat_candidate_videos"] == 2
    assert result["squat_evaluated_videos"] == 1
    assert result["excluded_videos"] == 1
    assert result["exclusion_reasons"] == {"ValueError": 1}
    assert result["event_recall"] == 0
