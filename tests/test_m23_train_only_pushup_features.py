"""M23 tests must not require media downloads or actual video decoders."""
from __future__ import annotations

from types import SimpleNamespace
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts import m23_train_only_pushup_features as m23


def landmarks():
    lm = [SimpleNamespace(x=.1 + i * .006, y=.2 + i * .003,
                          z=.01 * i, visibility=.95)
          for i in range(33)]
    for i in (11, 12):
        lm[i].x, lm[i].y = .4, .2
    for i in (23, 24):
        lm[i].x, lm[i].y = .4, .55
    return lm


def test_new_upper_schema_is_masked_and_fixed_width():
    v = landmarks()
    a = m23.upper32_frame(v)
    assert a.shape == (32,) and np.isfinite(a).all()
    # The former full-body gate excludes this frame; upper-only accepts it.
    for joint in (25, 26, 27, 28):
        v[joint].visibility = .01
    assert m23.diagnose_pushup_pose(v)[0] == "low_one_side_visibility"
    assert m23.upper32_frame(v).shape == (32,)
    # One invisible elbow in the worse side must be zero-masked.
    v[14].visibility = .1
    b = m23.upper32_frame(v).reshape(8, 4)
    assert np.all(b[3, :3] == 0)
    assert b[3, 3] == .1
    assert m23.aggregate_frame_vectors([a] * 3) is None
    assert m23.aggregate_frame_vectors([a] * 4).shape == (64,)
    assert m23.aggregate_frame_vectors([a] * 4).max() < 100
    v[0].x = float("nan")
    assert m23.upper32_frame(v) is None


def test_m23_frozen_train_only_subset_unchanged_after_recovery(monkeypatch):
    old = pd.DataFrame([
        {"video_id": "train_ok", "subject_id": "train_subject"},
        {"video_id": "val_ok", "subject_id": "validation_subject"},
        {"video_id": "test_ok", "subject_id": "test_subject"}
    ])
    monkeypatch.setattr(m23, "split_by_subject", lambda *_a, **_kw: {
        "train": old.iloc[[0]], "val": old.iloc[[1]], "test": old.iloc[[2]]
    })
    source = [
        {"video_id": "train_ok", "subject_id": "train_subject",
         "exercise": "pushup", "label": "good", "view": "side",
         "archive_member": "old.mp4"},
        {"video_id": "train_recovered", "subject_id": "train_subject",
         "exercise": "pushup", "label": "bad", "view": "front",
         "archive_member": "recovered.mp4"},
        {"video_id": "val_ok", "subject_id": "validation_subject",
         "exercise": "pushup", "label": "good", "view": "side",
         "archive_member": "val.mp4"},
        {"video_id": "val_recovered", "subject_id": "validation_subject",
         "exercise": "pushup", "label": "bad", "view": "front",
         "archive_member": "val_recovered.mp4"},
        {"video_id": "test_ok", "subject_id": "test_subject",
         "exercise": "pushup", "label": "good", "view": "side",
         "archive_member": "test.mp4"},
        {"video_id": "new_unknown", "subject_id": "not_in_original_m20",
         "exercise": "pushup", "label": "bad", "view": "front",
         "archive_member": "new.mp4"}
    ]
    selected, participants = m23.frozen_train_manifest(source, old)
    assert participants == {"train_subject"}
    assert {r["video_id"] for r in selected} == {"train_ok", "train_recovered"}
    broken = [dict(r) for r in source]
    broken[0]["subject_id"] = "validation_subject"
    with pytest.raises(ValueError, match="metadata differs"):
        m23.frozen_train_manifest(broken, old)
    with pytest.raises(ValueError, match="seed 42"):
        m23.frozen_train_manifest(source, old, seed=43)


def test_m23_does_not_open_validation_or_test_zip_video(tmp_path, monkeypatch):
    old = pd.DataFrame([
        {"video_id": "train_ok", "subject_id": "train_subject"},
        {"video_id": "val_ok", "subject_id": "validation_subject"},
        {"video_id": "test_ok", "subject_id": "test_subject"}
    ])
    monkeypatch.setattr(m23, "load_clip_table", lambda *_: old)
    monkeypatch.setattr(m23, "split_by_subject", lambda *_a, **_kw: {
        "train": old.iloc[[0]], "val": old.iloc[[1]], "test": old.iloc[[2]]
    })
    source = [
        {"video_id": "train_ok", "subject_id": "train_subject",
         "exercise": "pushup", "label": "good", "view": "front",
         "archive_member": "train.mp4"},
        {"video_id": "val_ok", "subject_id": "validation_subject",
         "exercise": "pushup", "label": "bad", "view": "front",
         "archive_member": "val.mp4"},
        {"video_id": "test_ok", "subject_id": "test_subject",
         "exercise": "pushup", "label": "good", "view": "front",
         "archive_member": "test.mp4"}
    ]
    monkeypatch.setattr(m23, "read_manifest", lambda *_: source)

    class Pose:
        def __enter__(self): return self
        def __exit__(self, *_): return False

    class FakeMP:
        solutions = SimpleNamespace(pose=SimpleNamespace(
            Pose=lambda **_kwargs: Pose()
        ))
    monkeypatch.setitem(sys.modules, "mediapipe", FakeMP())
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())
    inspected = []

    def fake_extract(_archive, member, *_args, **_kwargs):
        inspected.append(member.filename)
        return {"clip_vectors": {
            "full_body": np.zeros(64),
            "upper_body": np.ones(64)
        }, "accepted_frame_counts": {
            "full_body": 4, "upper_body": 5
        }, "sampled": 6}
    monkeypatch.setattr(m23, "_extract_video", fake_extract)
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in ("train.mp4", "val.mp4", "test.mp4"):
            z.writestr(name, b"synthetic")
    dest = tmp_path / "research"
    result = m23.extract_train(archive, tmp_path/"manifest.csv",
                               tmp_path/"original.csv", dest)
    assert inspected == ["train.mp4"]
    assert result["validation_or_test_video_decoded"] is False
    assert result["gate_counts"]["upper_body"]["accepted_clips"] == 1
    actual = pd.read_csv(dest/"train_upper_body.csv")
    assert len(actual) == 1
    assert actual.shape[1] == len(m23.COLS)
    with pytest.raises(ValueError, match="Output already exists"):
        m23.extract_train(archive, tmp_path/"manifest.csv",
                          tmp_path/"original.csv", dest)
