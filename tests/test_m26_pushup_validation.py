"""M26 synthetic proof that VAL-only decoding preserves TEST and cohort matching."""
from __future__ import annotations

import json
from types import SimpleNamespace
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts import m26_pushup_validation as m26
from scripts.m23_train_only_pushup_features import COLS, FEATURE_COLUMNS


def sample(name, person, label, view, *, exercise="pushup"):
    return {
        "video_id": name, "subject_id": person, "label": label,
        "view": view, "exercise": exercise, "archive_member": name + ".mp4",
    }


def source_dataset():
    # TRAIN and original VAL/TEST participants must never share a group.
    rows = [
        sample("train_good", "a_train", "good", "front"),
        sample("train_bad", "a_train", "bad", "side"),
        sample("train_recovered", "a_train", "bad", "front"),
        sample("val_good", "b_val", "good", "diagonal"),
        sample("val_bad", "b_val", "bad", "front"),
        sample("val_recovered", "b_val", "bad", "front"),
        sample("test_good", "c_test", "good", "front"),
        sample("test_bad", "c_test", "bad", "side"),
    ]
    original = pd.DataFrame([x for x in rows if x["video_id"] not in
                             ("train_recovered", "val_recovered")])
    return rows, original


def stub_original_split(monkeypatch):
    monkeypatch.setattr(m26, "split_by_subject", lambda df, **_k: {
        "train": df[df.subject_id == "a_train"],
        "val": df[df.subject_id == "b_val"],
        "test": df[df.subject_id == "c_test"]
    })


def table(rows):
    examples = []
    for i, r in enumerate(rows):
        example = {k: r[k] for k in ("video_id", "subject_id",
                                    "exercise", "label", "view")}
        example.update({f: float(i + j / 100)
                        for j, f in enumerate(FEATURE_COLUMNS)})
        examples.append(example)
    return pd.DataFrame(examples, columns=COLS).set_index("video_id", drop=False)


def test_m26_frozen_validation_split_original_only(monkeypatch):
    manifest, original = source_dataset()
    stub_original_split(monkeypatch)
    people, rows = m26.frozen_splits(manifest, original)
    assert {r["video_id"] for r in rows["val"]} == {
        "val_good", "val_bad", "val_recovered"
    }
    assert "test_good" not in {r["video_id"] for r in rows["val"]}
    assert people["val"] == {"b_val"}
    altered = [dict(x) for x in manifest]
    altered[0]["subject_id"] = "b_val"
    with pytest.raises(ValueError, match="metadata disagrees"):
        m26.frozen_splits(altered, original)


def test_m26_decode_only_val_videos_never_test_or_train(monkeypatch, tmp_path):
    manifest, original = source_dataset()
    stub_original_split(monkeypatch)
    monkeypatch.setattr(m26, "read_manifest", lambda _: manifest)
    monkeypatch.setattr(m26, "load_clip_table", lambda _: original)
    class Pose:
        def __enter__(self): return self
        def __exit__(self, *_): return False
    mp = SimpleNamespace(solutions=SimpleNamespace(
        pose=SimpleNamespace(Pose=lambda **_kwargs: Pose())))
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())
    decoded = []

    def fake_extract(_zipped, member, *_args, **_kwargs):
        decoded.append(member.filename)
        return {
            "clip_vectors": {
                "full_body": None if member.filename == "val_recovered.mp4"
                    else np.ones(64),
                "upper_body": np.ones(64),
            },
            "accepted_frame_counts": {"full_body": 4, "upper_body": 5},
            "sampled": 10,
        }
    monkeypatch.setattr(m26, "_extract_video", fake_extract)
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for row in manifest:
            zipped.writestr(row["archive_member"], b"synthetic")
    destination = tmp_path / "m26_val"
    result = m26.extract_validation(
        archive, tmp_path/"manifest.csv", tmp_path/"original.csv",
        destination
    )
    assert sorted(decoded) == [
        "val_bad.mp4", "val_good.mp4", "val_recovered.mp4"
    ]
    assert result["original_TEST_video_decoded"] is False
    assert result["original_TRAIN_video_decoded"] is False
    assert result["validation_manifest_clips"] == 3
    assert result["full_gate_accepted"] == 2
    assert result["upper_gate_accepted"] == 3
    assert list(pd.read_csv(destination/"val_full_body.csv")["video_id"]) == [
        "val_bad", "val_good"
    ]
    assert "a_train" not in json.dumps(result)
    assert "b_val" not in json.dumps(result)
    with pytest.raises(FileExistsError):
        m26.extract_validation(
            archive, tmp_path/"manifest.csv", tmp_path/"original.csv",
            destination
        )


def test_m26_matched_scoring_and_unpaired_extra_never_leak_test(monkeypatch, tmp_path):
    manifest, original = source_dataset()
    stub_original_split(monkeypatch)
    monkeypatch.setattr(m26, "read_manifest", lambda _: manifest)
    monkeypatch.setattr(m26, "load_clip_table", lambda _: original)

    source = {r["video_id"]: r for r in manifest}
    def path(name, ids):
        output = tmp_path / name
        table([source[i] for i in ids]).to_csv(output, index=False)
        return output

    full_train = path("train_full.csv", ["train_good", "train_bad"])
    upper_train = path("train_upper.csv",
                       ["train_bad", "train_recovered", "train_good"])
    full_val = path("val_full.csv", ["val_good", "val_bad"])
    upper_val = path("val_upper.csv",
                     ["val_recovered", "val_bad", "val_good"])

    fit_calls, predicted = [], []
    def fake_fit(frame, *, n_trees):
        fit_calls.append((set(frame.video_id), n_trees))
        assert set(frame.subject_id) == {"a_train"}
        return object()
    def fake_predict(_model, frame):
        predicted.append(list(frame.video_id))
        assert set(frame.subject_id) == {"b_val"}
        return np.asarray([.8 if label == "bad" else .2 for label in frame.label])
    monkeypatch.setattr(m26, "_fit", fake_fit)
    monkeypatch.setattr(m26, "_predict", fake_predict)

    report = m26.evaluate_validation(
        tmp_path/"original.csv", tmp_path/"manifest.csv",
        full_train, upper_train, full_val, upper_val
    )
    assert report["original_VAL_pushup_clips"] == 3
    assert report["paired_common_val_clips"] == 2
    assert report["val_upper_only_recovered"] == 1
    assert report["original_TEST_scored"] is False
    assert report["original_TEST_video_decoded"] is False
    assert set(report["paired_model_scores"]) == set(m26.TRAIN_SETS)
    assert all(x["all_common"]["n"] == 2 for x in
               report["paired_model_scores"].values())
    assert report["unpaired_recovered_val_clips"]["all_views"]["n"] == 1
    assert [names for names, trees in fit_calls] == [
        ({"train_good", "train_bad"}, 200),
        ({"train_good", "train_bad", "train_recovered"}, 200),
    ]
    assert predicted == [
        ["val_bad", "val_good"], ["val_bad", "val_good"], ["val_recovered"]
    ]
    assert "c_test" not in json.dumps(report)
    # Any TEST identity injected into an original validation table must fail.
    invalid = path("invalid.csv", ["val_good", "test_bad"])
    with pytest.raises(ValueError, match="Full-gate|non-VAL"):
        m26.evaluate_validation(
            tmp_path/"original.csv", tmp_path/"manifest.csv",
            full_train, upper_train, invalid, upper_val
        )
