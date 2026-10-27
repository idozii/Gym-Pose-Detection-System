"""M21 data-quality diagnosis tests use synthetic landmark arrays / ZIPs only."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import zipfile

import pandas as pd
import pytest

from scripts import m21_audit_front_pose as m21


def pose(*, visibility=1., xy_gap=.3):
    points = [
        SimpleNamespace(x=.20 + i * .002, y=.10 + i * .005, z=-.1, visibility=visibility)
        for i in range(33)
    ]
    # Force meaningful shoulder/hip center separation regardless of sequential IDs.
    for i in (11, 12):
        points[i].x, points[i].y = .3, .2
    for i in (23, 24):
        points[i].x, points[i].y = .3, .2 + xy_gap
    return points


def test_m21_frame_reason_matches_original_m20_gate():
    good = pose()
    assert m21.diagnose_pushup_pose(good)[0] == "accepted"
    assert m21.diagnose_pushup_pose(None)[0] == "no_pose"
    assert m21.diagnose_pushup_pose(good[:32])[0] == "no_pose"
    tiny = pose(xy_gap=.01)
    assert m21.diagnose_pushup_pose(tiny)[0] == "low_torso_scale"
    low_all = pose()
    for i in (11, 12):
        low_all[i].visibility = .05
    category, missing = m21.diagnose_pushup_pose(low_all)
    assert category == "low_one_side_visibility"
    assert set(missing) == {"left_shoulder", "right_shoulder"}
    one_side = pose()
    for i in (11, 13, 15, 23, 25, 27):
        one_side[i].visibility = .05
    assert m21.diagnose_pushup_pose(one_side)[0] == "accepted"
    nan_pose = pose()
    nan_pose[2].x = float("nan")
    assert m21.diagnose_pushup_pose(nan_pose)[0] == "invalid_landmarks"


def test_m21_audit_does_not_access_holdout_val_or_test_subjects(tmp_path, monkeypatch):
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in ("train.mp4", "val.mp4", "test.mp4"):
            z.writestr(name, b"synthetic data, never decoded")
    frame = pd.DataFrame([
        {"video_id": "oktrain", "subject_id": "train_person"},
        {"video_id": "okval", "subject_id": "val_person"},
        {"video_id": "oktest", "subject_id": "test_person"},
    ])
    monkeypatch.setattr(m21, "load_clip_table", lambda _: frame)
    monkeypatch.setattr(m21, "split_by_subject", lambda *_args, **_kwargs: {
        "train": frame.iloc[[0]], "val": frame.iloc[[1]], "test": frame.iloc[[2]]
    })
    rows = [
        {"video_id": "badtrain", "archive_member": "train.mp4",
         "subject_id": "train_person", "exercise": "pushup", "view": "front", "label": "bad"},
        {"video_id": "badval", "archive_member": "val.mp4",
         "subject_id": "val_person", "exercise": "pushup", "view": "front", "label": "bad"},
        {"video_id": "badtest", "archive_member": "test.mp4",
         "subject_id": "test_person", "exercise": "pushup", "view": "front", "label": "bad"},
        {"video_id": "oktrain", "archive_member": "already.mp4",
         "subject_id": "train_person", "exercise": "bicep_curl", "view": "side", "label": "good"},
        {"video_id": "okval", "archive_member": "another.mp4",
         "subject_id": "val_person", "exercise": "bicep_curl", "view": "side", "label": "good"},
        {"video_id": "oktest", "archive_member": "final.mp4",
         "subject_id": "test_person", "exercise": "bicep_curl", "view": "side", "label": "good"},
    ]
    monkeypatch.setattr(m21, "read_manifest", lambda _: rows)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace())
    inspected = []
    def fake_inspect(entry, *_args, **_kwargs):
        inspected.append(entry.filename)
        return {"frames": 10, "accepted": 0,
                "reasons": {"low_one_side_visibility": 10},
                "missing_joints": {"left_ankle": 10}}
    monkeypatch.setattr(m21, "_inspect_one", fake_inspect)
    output = tmp_path / "ignored_report.json"
    summary = m21.audit(archive, tmp_path/"manifest.csv",
                        tmp_path/"m20.csv", output)
    assert inspected == ["train.mp4"]
    assert summary["validation_or_test_decoded"] is False
    assert summary["selected_train_rejected_clips"] == 1
    assert summary["by_label"]["bad"]["missing_joints"]["left_ankle"] == 10
    assert output.is_file()
    with pytest.raises(FileExistsError):
        m21.audit(archive, tmp_path/"manifest.csv",
                  tmp_path/"m20.csv", output)


def test_m21_invalid_limits_fail_without_decoding(tmp_path):
    with pytest.raises(ValueError, match="Invalid diagnostic limits"):
        m21.audit(tmp_path/"x.zip", tmp_path/"y.csv", tmp_path/"z.csv",
                  tmp_path/"out.json", max_clips=-1)
