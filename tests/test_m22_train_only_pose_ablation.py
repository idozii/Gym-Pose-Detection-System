"""M22 synthetic tests: controlled gates, unchanged train/holdout split."""
from __future__ import annotations

from types import SimpleNamespace
import sys
import zipfile

import pandas as pd
import pytest

from scripts import m22_train_only_pose_ablation as m22


def points():
    arr = [SimpleNamespace(x=.45 + i * .001, y=.32 + i * .001,
                           z=.05, visibility=.95) for i in range(33)]
    for k in (11, 12):
        arr[k].x, arr[k].y = .4, .20
    for k in (23, 24):
        arr[k].x, arr[k].y = .4, .55
    return arr


def test_upper_body_gate_retains_raw_landmark_units_and_does_not_require_legs():
    p = points()
    assert m22.upper_body_gate(p) == "accepted"
    assert m22.diagnose_pushup_pose(p)[0] == "accepted"
    for i in (25, 26, 27, 28):
        p[i].visibility = .1
    assert m22.diagnose_pushup_pose(p)[0] == "low_one_side_visibility"
    assert m22.upper_body_gate(p) == "accepted"
    for i in (13, 14):
        p[i].visibility = .1
    assert m22.upper_body_gate(p) == "low_upper_body_visibility"
    assert m22.upper_body_gate(None) == "no_pose"
    assert m22.upper_body_gate(p[:30]) == "no_pose"
    x = points()
    x[0].x = float("nan")
    assert m22.upper_body_gate(x) == "invalid_landmarks"
    x = points()
    for i in (23, 24):
        x[i].y = .21
    assert m22.upper_body_gate(x) == "low_torso_scale"


def test_subject_holdouts_and_already_extracted_videos_are_not_selected(monkeypatch):
    frame = pd.DataFrame([
        {"video_id": "oktrain", "subject_id": "train_a"},
        {"video_id": "okval", "subject_id": "val_b"},
        {"video_id": "oktest", "subject_id": "test_c"}
    ])
    monkeypatch.setattr(m22, "split_by_subject", lambda *_a, **_k: {
        "train": frame.iloc[[0]], "val": frame.iloc[[1]],
        "test": frame.iloc[[2]]
    })
    rows = [
        {"video_id": "new_train", "subject_id": "train_a",
         "archive_member": "a.mp4", "exercise": "pushup", "view": "front",
         "label": "bad"},
        {"video_id": "new_val", "subject_id": "val_b",
         "archive_member": "b.mp4", "exercise": "pushup", "view": "front",
         "label": "bad"},
        {"video_id": "new_test", "subject_id": "test_c",
         "archive_member": "c.mp4", "exercise": "pushup", "view": "front",
         "label": "good"},
        {"video_id": "oktrain", "subject_id": "train_a",
         "archive_member": "d.mp4", "exercise": "pushup", "view": "front",
         "label": "good"},
        {"video_id": "okval", "subject_id": "val_b",
         "archive_member": "e.mp4", "exercise": "bicep_curl", "view": "side",
         "label": "bad"},
        {"video_id": "oktest", "subject_id": "test_c",
         "archive_member": "f.mp4", "exercise": "bicep_curl", "view": "side",
         "label": "bad"}
    ]
    assert [r["video_id"] for r in m22.select_train_rejected_front(rows, frame)] == [
        "new_train"
    ]
    broken = [dict(row) for row in rows]
    broken[-1]["subject_id"] = "train_a"
    with pytest.raises(ValueError, match="differ"):
        m22.select_train_rejected_front(broken, frame)


def test_full_audit_reads_only_train_video_and_generates_aggregate_json(
        tmp_path, monkeypatch):
    frame = pd.DataFrame([
        {"video_id": "oktrain", "subject_id": "train_a"},
        {"video_id": "okval", "subject_id": "val_b"},
        {"video_id": "oktest", "subject_id": "test_c"}
    ])
    monkeypatch.setattr(m22, "load_clip_table", lambda _path: frame)
    monkeypatch.setattr(m22, "split_by_subject", lambda *_a, **_k: {
        "train": frame.iloc[[0]], "val": frame.iloc[[1]],
        "test": frame.iloc[[2]]
    })
    rows = [
        {"video_id": "train_bad", "subject_id": "train_a", "exercise": "pushup",
         "view": "front", "label": "bad", "archive_member": "train.mp4"},
        {"video_id": "val_bad", "subject_id": "val_b", "exercise": "pushup",
         "view": "front", "label": "bad", "archive_member": "val.mp4"},
        {"video_id": "test_good", "subject_id": "test_c", "exercise": "pushup",
         "view": "front", "label": "good", "archive_member": "test.mp4"},
        {"video_id": "oktrain", "subject_id": "train_a", "exercise": "pushup",
         "view": "side", "label": "good", "archive_member": "ok.mp4"},
        {"video_id": "okval", "subject_id": "val_b", "exercise": "pushup",
         "view": "side", "label": "good", "archive_member": "okval.mp4"},
        {"video_id": "oktest", "subject_id": "test_c", "exercise": "pushup",
         "view": "side", "label": "good", "archive_member": "oktest.mp4"}
    ]
    monkeypatch.setattr(m22, "read_manifest", lambda _path: rows)
    class FakeDetector:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
    class FakeMP:
        solutions = SimpleNamespace(pose=SimpleNamespace(
            Pose=lambda **_args: FakeDetector()
        ))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mediapipe", FakeMP())
    examined = []
    def fake_inspect(entry, *_args, **_kwargs):
        examined.append(entry.filename)
        return {
            "sampled": 6,
            "modes": {name: {
                gate: {"accepted": 4 if gate == "upper_body" else 2,
                       "low_one_side_visibility": 4 if gate == "full_body" else 0,
                       "low_upper_body_visibility": 2 if gate == "upper_body" else 0}
                for gate in m22.GATES
            } for name in m22.DETECTORS}
        }
    monkeypatch.setattr(m22, "_inspect_video", fake_inspect)
    archive = tmp_path / "research.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for n in ("train.mp4", "val.mp4", "test.mp4"):
            z.writestr(n, b"fake")
    out = tmp_path / "audit.json"
    result = m22.audit(archive, tmp_path/"manifest.csv",
                       tmp_path/"features.csv", out)
    assert examined == ["train.mp4"]
    assert result["validation_or_test_decoded"] is False
    assert result["selected_train_rejected_clips"] == 1
    assert result["comparisons"]["baseline"]["full_body"]["total"][
        "accepted_frames"] == 2
    assert result["comparisons"]["baseline"]["upper_body"]["total"][
        "clips_reaching_four_frames"] == 1
    text = out.read_text()
    assert "train_a" not in text and "train.mp4" not in text
    with pytest.raises(FileExistsError):
        m22.audit(archive, tmp_path/"manifest.csv", tmp_path/"features.csv", out)


def test_wrong_split_seed_and_invalid_controls_rejected_without_source_access(
        tmp_path):
    with pytest.raises(ValueError, match="Invalid"):
        m22.audit(tmp_path/"a", tmp_path/"m", tmp_path/"f",
                  tmp_path/"out", seed=43)
