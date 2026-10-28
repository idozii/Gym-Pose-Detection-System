"""M50 offline synthetic publisher TRAIN CSV and OpenCV timing tests."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts import m50_train_label_video_integrity as m50


def _mock_csv(*, extra_subject=True):
    names = [f"train_squat_{n:03d}.mp4" for n in range(40)]
    salt = f"repcount-m40-train-v1:{m50.SEED}:".encode("ascii")
    ranked = sorted(names, key=lambda name: (
        hashlib.sha256(salt + name.encode("ascii")).digest(), name
    ))
    ranks = {name:i for i,name in enumerate(ranked)}
    cols = ["name", "type", "count"] + [
        f"L{i}" for i in range(1, 55)
    ]
    if extra_subject:
        cols.extend(["subject_id", "session_id"])
    writer_file = io.StringIO()
    writer = csv.DictWriter(writer_file, fieldnames=cols)
    writer.writeheader()
    for name in reversed(names):
        rank = ranks[name]
        count = (27 if rank == 0 else 15) if rank < 20 else (
            19 if rank == 20 else 10
        )
        record = {"name":name, "type":"squat", "count":count}
        for i in range(count):
            record[f"L{2*i+1}"] = i*6
            record[f"L{2*i+2}"] = i*6+4
        if extra_subject:
            # Two files in different cohorts explicitly share one
            # synthetic subject identifier; NEVER store raw identifier.
            record["subject_id"] = (
                "shared-person" if rank in (0,20)
                else f"person-{rank:03d}"
            )
            record["session_id"] = f"session-{rank:03d}"
        writer.writerow(record)
    return writer_file.getvalue().encode("utf8")


def _local_sources(tmp_path, cohorts):
    roots = []
    for i, cohort in enumerate(cohorts):
        root = tmp_path / ("development" if i == 0 else "replication")
        video_dir = root / "official_tar_selected" / "train"
        video_dir.mkdir(parents=True)
        for row in cohort:
            (video_dir/row["video"]).write_bytes(b"mock")
        roots.append(root)
    return roots


class VideoCap:
    def __init__(self, *, fps=25, declared=100, actual=102):
        self.fps = fps
        self.declared = declared
        self.actual = actual
        self.count = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, code):
        return self.fps if code == 5 else self.declared

    def read(self):
        if self.count >= self.actual:
            return False, None
        self.count += 1
        return True, None

    def release(self):
        self.released = True


class CV2:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7

    def __init__(self, **args):
        self.args = args
        self.cap = None

    def VideoCapture(self, _name):
        self.cap = VideoCap(**self.args)
        return self.cap


def test_m50_probe_decodes_entire_source_not_metadata_prefix(tmp_path):
    p = tmp_path/"mock.mp4"
    p.write_bytes(b"x"*12000)
    cv = CV2(fps=25, declared=100, actual=110)
    result = m50.probe_video(p, cv2_module=cv)
    assert result == {"fps":25., "claimed_frames":100,
                      "decoded_frames":110}
    assert cv.cap.released is True
    broken = CV2(fps=float("nan"), declared=100, actual=100)
    with pytest.raises(ValueError, match="metadata invalid"):
        m50.probe_video(p, cv2_module=broken)
    assert broken.cap.released


def test_m50_detects_bounds_past_actual_video_without_guessing_label_alignment():
    ann = {
        "annotated_count":2, "cycle_bounds_frames":[0,10,13,20]
    }
    result = m50._video_audit(
        ann, {"fps":10., "claimed_frames":24, "decoded_frames":19}
    )
    assert result["label_bounds_within_decoded_video"] is False
    assert result["label_bounds_within_claimed_video"] is True
    assert result["frame_count_difference"] == -5
    assert result["short_gt_intervals_lt_0_4_sec"] == 0
    assert result["minimum_between_rep_gap_seconds"] == pytest.approx(.3)


def test_m50_source_rank_and_subject_metadata_remain_strictly_train_only():
    raw = _mock_csv()
    cohorts = m50._publisher_cohorts(raw)
    assert [sum(r["annotated_count"] for r in cohort) for cohort in cohorts] == [
        312,209
    ]
    names = [[row["video"].lower() for row in cohort] for cohort in cohorts]
    assert set(names[0]).isdisjoint(names[1])
    metadata = m50._header_metadata(raw,names)
    assert metadata["subject"]["recognized_explicit_column"] is True
    assert metadata["subject"]["all_40_source_rows_populated"] is True
    assert metadata["subject"]["observed_cross_cohort_shared_ids"] == 1
    assert metadata["subject"]["across_cohort_id_overlap_verified"] is False
    assert metadata["camera"]["recognized_explicit_column"] is False
    assert "shared-person" not in json.dumps(metadata)
    assert "person-000" not in json.dumps(metadata)
    without = _mock_csv(extra_subject=False)
    records = m50._publisher_cohorts(without)
    m = m50._header_metadata(without, [
        [e["video"].lower() for e in r] for r in records
    ])
    assert m["subject"]["observed_distinct_ids"] is None
    assert m["subject"]["across_cohort_id_overlap_verified"] is False


def test_m50_aggregates_40_source_train_clips_but_retains_no_private_fields(tmp_path):
    raw = _mock_csv()
    archive = tmp_path/"RepCountA.tar.gz"
    archive.write_bytes(b"synthetic-stand-in-for-tar")
    cohorts = m50._publisher_cohorts(raw)
    dev, repl = _local_sources(tmp_path, cohorts)
    lock = tmp_path/"m48.json"
    lock.write_text(json.dumps({
        "publisher_train_annotation_sha256":hashlib.sha256(raw).hexdigest()
    }))
    report = tmp_path/"report.json"
    calls = []
    def fake_probe(path):
        calls.append(path.parent.parent.name)
        return {"fps":30., "claimed_frames":1000, "decoded_frames":1000}
    def gate(**kwargs):
        assert kwargs["check_only"] is True
        return {"research_source_provenance_check_passed":True}
    r = m50.run(
        archive=archive, development_root=dev, replication_root=repl,
        m46_path=tmp_path/"m46.json", m47_path=tmp_path/"m47.json",
        m48_lock=lock, report=report,
        source_fn=lambda path,split:raw if split == "train" else None,
        probe_fn=fake_probe, freeze_gate=gate
    )
    assert len(calls) == 40
    assert r["videos_total"] == 40
    assert r["annotated_gt_reps_total"] == 521
    assert r["critical_timing_or_decode_integrity_flags"] == 0
    assert r["cohort_video_timing"][0]["annotated_reps"] == 312
    assert r["cohort_video_timing"][1]["annotated_reps"] == 209
    assert r["validation_or_test_scanned"] is False
    assert r["detector_or_matcher_modified"] is False
    assert r["explicit_csv_identifier_availability"]["subject"][
        "observed_cross_cohort_shared_ids"
    ] == 1
    text = report.read_text()
    for banned in ("shared-person", "person-001", "train_squat_000",
                   "landmarks", '"frames"', '"video_path"',
                   "cycle_bounds_frames"):
        assert banned not in text
    with pytest.raises(FileExistsError,match="refusing overwrite"):
        m50.run(archive=archive, development_root=dev, replication_root=repl,
                m46_path=tmp_path/"m46.json", m47_path=tmp_path/"m47.json",
                m48_lock=lock, report=report)


def test_m50_mismatched_source_sha_and_missing_cohort_fail_closed(tmp_path):
    raw = _mock_csv()
    archive = tmp_path/"RepCountA.tar.gz"
    archive.write_bytes(b"x")
    cohorts = m50._publisher_cohorts(raw)
    dev, repl = _local_sources(tmp_path,cohorts)
    lock = tmp_path/"m48.json"
    lock.write_text(json.dumps({"publisher_train_annotation_sha256":"0"*64}))
    kwargs = dict(
        archive=archive, development_root=dev, replication_root=repl,
        m46_path=tmp_path/"m46.json",m47_path=tmp_path/"m47.json",
        m48_lock=lock, report=tmp_path/"report.json",
        source_fn=lambda _a,_s:raw,
        probe_fn=lambda path:{"fps":30.,"claimed_frames":1000,"decoded_frames":1000},
        freeze_gate=lambda **kw:{"research_source_provenance_check_passed":True}
    )
    with pytest.raises(ValueError,match="CSV SHA256 differs"):
        m50.run(**kwargs)
    lock.write_text(json.dumps({
        "publisher_train_annotation_sha256":hashlib.sha256(raw).hexdigest()
    }))
    (repl/"official_tar_selected"/"train"/cohorts[1][0]["video"]).unlink()
    with pytest.raises(ValueError,match="TRAIN filenames differ"):
        m50.run(**kwargs)
    assert not (tmp_path/"report.json").exists()


def test_m50_anomalous_decoded_video_bounds_are_counted_not_hidden(tmp_path):
    raw = _mock_csv()
    archive = tmp_path/"RepCountA.tar.gz"
    archive.write_bytes(b"x")
    cohorts = m50._publisher_cohorts(raw)
    dev, repl = _local_sources(tmp_path,cohorts)
    lock = tmp_path/"m48.json"
    lock.write_text(json.dumps({
        "publisher_train_annotation_sha256":hashlib.sha256(raw).hexdigest()
    }))
    seq = [0]
    def fake_probe(path):
        seq[0] += 1
        if seq[0] == 1:
            return {"fps":25.,"claimed_frames":1000,"decoded_frames":25}
        return {"fps":25.,"claimed_frames":1000,"decoded_frames":1000}
    result = m50.run(
        archive=archive, development_root=dev, replication_root=repl,
        m46_path=tmp_path/"m46.json",m47_path=tmp_path/"m47.json",
        m48_lock=lock,report=tmp_path/"report.json",
        source_fn=lambda _a,_s:raw,probe_fn=fake_probe,
        freeze_gate=lambda **kw:{"research_source_provenance_check_passed":True}
    )
    assert result["critical_timing_or_decode_integrity_flags"] >= 1
    assert result["cohort_video_timing"][0][
        "videos_decoding_below_90pct_metadata"
    ] == 1
    assert result["cohort_video_timing"][0][
        "videos_with_gt_bounds_exceeding_decoded_frames"
    ] == 1
