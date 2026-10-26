"""Task-specific OHP input features and source-data safety (synthetic only)."""
from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gym_pose.ohp_task_features import (
    FEATURES_BY_TASK, SCHEMA_BY_TASK, TASKS, pose_to_task_features,
)
from scripts import extract_fitness_aqa_ohp_tasks as prep


def landmarks(low=(), bad_unneeded=False):
    items = [
        SimpleNamespace(x=.02 * i, y=.03 * i, z=.001 * i,
                        visibility=(.3 if i in low else .99))
        for i in range(33)
    ]
    if bad_unneeded:
        items[0].x = float("nan")
    return items


def test_exact_task_only_joint_visibility_and_stable_30_feature_schema():
    for task in TASKS:
        result = pose_to_task_features(landmarks(), task=task)
        assert result is not None
        assert len(result) == 30
        assert tuple(result) == FEATURES_BY_TASK[task]
        assert len(set(result)) == 30
        assert SCHEMA_BY_TASK[task].startswith("ohp-")
        assert all(np.isfinite(list(result.values())))
        assert pose_to_task_features(landmarks(bad_unneeded=True), task=task) is not None
    lower_not_trusted = landmarks(low=(25, 27))
    assert pose_to_task_features(lower_not_trusted, task="error_elbows") is not None
    assert pose_to_task_features(lower_not_trusted, task="error_knees") is None
    upper_not_trusted = landmarks(low=(13, 15))
    assert pose_to_task_features(upper_not_trusted, task="error_knees") is not None
    assert pose_to_task_features(upper_not_trusted, task="error_elbows") is None
    assert pose_to_task_features(landmarks(low=(11,)), task="error_elbows") is None
    assert pose_to_task_features(landmarks(low=(11,)), task="error_knees") is None


def test_untrustworthy_geometry_or_invalid_arguments_abstains():
    assert pose_to_task_features(None, task="error_elbows") is None
    assert pose_to_task_features(landmarks()[:32], task="error_knees") is None
    degenerate = landmarks()
    degenerate[11].x = degenerate[12].x
    degenerate[11].y = degenerate[12].y
    degenerate[11].z = degenerate[12].z
    assert pose_to_task_features(degenerate, task="error_knees") is None
    with pytest.raises(ValueError, match="Unknown OHP"):
        pose_to_task_features(landmarks(), task="error_undefined")
    with pytest.raises(ValueError, match="min_visibility"):
        pose_to_task_features(landmarks(), task="error_elbows", min_visibility=float("nan"))


def source_fixture(tmp_path):
    folder = tmp_path / "ohp" / "labeled"
    ann = folder / "annotations"
    ann.mkdir(parents=True)
    splits = {"train": ["101_1", "102_1"], "val": ["201_1"], "test": ["999_1"]}
    for split, vids in splits.items():
        (ann / f"{split}_keys.json").write_text(json.dumps(vids))
    (ann / "error_elbows.json").write_text(json.dumps({
        "101_1": [[.35,.65],[.05,.12]], "102_1": [], "201_1":[[.35,.65]],
        "999_1": []}))
    (ann / "error_knees.json").write_text(json.dumps({
        "101_1": [], "102_1": [[.35,.65]], "201_1": [[.35,.65]],
        "999_1": []}))
    with zipfile.ZipFile(folder / "videos.zip", "w") as archive:
        for vid in ["101_1","102_1","201_1"]:
            archive.writestr(f"videos/{vid}.mp4", b"synthetic")
        archive.writestr("videos/999_1.mp4", b"TEST SHOULD NOT DECODE")
    return folder


class Cap:
    def __init__(self):
        self.count = 0
    def isOpened(self):
        return True
    def get(self, _):
        return 10.
    def read(self):
        if self.count >= 6:
            return False, None
        self.count += 1
        return True, np.zeros((4,4,3),dtype=np.uint8)
    def release(self):
        pass


class CV2:
    CAP_PROP_FPS = 5
    COLOR_BGR2RGB = 6
    @staticmethod
    def VideoCapture(path):
        assert str(path).endswith(".mp4")
        return Cap()
    @staticmethod
    def cvtColor(frame, code):
        return frame


class MPose:
    def __init__(self, **kwargs):
        assert kwargs["static_image_mode"] is True
        self.counter = 0
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def process(self, frame):
        self.counter += 1
        if self.counter % 3 == 1:
            points = landmarks(low=(25,))
        elif self.counter % 3 == 2:
            points = landmarks(low=(15,))
        else:
            points = landmarks()
        return SimpleNamespace(
            pose_landmarks=SimpleNamespace(landmark=points))


MP = SimpleNamespace(solutions=SimpleNamespace(
    pose=SimpleNamespace(Pose=MPose)))


def test_extract_two_different_validated_csvs_without_test_or_legacy_overwrite(tmp_path):
    source = source_fixture(tmp_path)
    out = tmp_path / "features"
    out.mkdir()
    legacy = out / "fitness_aqa_ohp_train_pose.csv"
    legacy.write_text("LEGACY MUST REMAIN UNCHANGED")
    report = prep.extract(source / "videos.zip", source / "annotations",
                          out, max_videos_per_split=0,
                          boundary_margin=0, sample_fps=2.,
                          cv2_api=CV2, mp_api=MP)
    assert report["source_test_videos_processed"] == 0
    assert report["splits"]["train"]["selected_videos"] == 2
    assert report["splits"]["val"]["selected_videos"] == 1
    assert report["splits"]["train"]["eligible_frames"] == 4
    assert report["splits"]["val"]["eligible_frames"] == 2
    assert report["splits"]["train"]["videos_failed"] == 0
    assert legacy.read_text() == "LEGACY MUST REMAIN UNCHANGED"
    for split in ("train","val"):
        totals = report["splits"][split]
        for task in TASKS:
            file = out / f"fitness_aqa_ohp_{task}_{split}_pose.csv"
            df = pd.read_csv(file)
            assert len(df) == totals["tasks"][task]["accepted_frames"]
            assert list(df) == list(prep.OUTPUT_COLUMNS[task])
            assert (df["split"] == split).all()
            assert not df.sample_id.duplicated().any()
            assert set(df["video_id"]) <= (
                {"101_1","102_1"} if split=="train" else {"201_1"})
            expected = {
                "train": {"error_elbows": 3, "error_knees": 2},
                "val": {"error_elbows": 1, "error_knees": 2},
            }
            assert totals["tasks"][task]["accepted_frames"] == expected[split][task]
            assert totals["tasks"][task]["positive_frames"] <= expected[split][task]
    assert not (out / "fitness_aqa_ohp_error_elbows_test_pose.csv").exists()
    assert not (out / "fitness_aqa_ohp_error_knees_test_pose.csv").exists()


def test_test_selection_refused_before_opening_source(tmp_path):
    with pytest.raises(ValueError, match="TEST locked"):
        prep.extract(tmp_path,tmp_path,tmp_path,selected_splits=("test",))
    with pytest.raises(ValueError, match="TEST locked"):
        prep.extract(tmp_path,tmp_path,tmp_path,selected_splits=("train","test"))


def test_bad_options_rejected_before_media_open(tmp_path):
    with pytest.raises(ValueError):
        prep.extract(tmp_path,tmp_path,tmp_path, sample_fps=0)
    with pytest.raises(ValueError):
        prep.extract(tmp_path,tmp_path,tmp_path, min_visibility=float("nan"))
