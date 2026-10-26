"""No download, MediaPipe, training or source test video needed for these tests."""
import csv
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gym_pose.features import FEATURE_NAMES
from gym_pose.ohp_features import (
    OHP_FEATURE_NAMES, OHP_EXTRA_NAMES, OHP_SCHEMA_VERSION, pose_to_ohp_features,
)
from scripts import extract_fitness_aqa_ohp as ohp


def landmarks(*, elbows_visibility=1.):
    items = [SimpleNamespace(x=.02 * i, y=.03 * i, z=.01 * i,
                             visibility=1.) for i in range(33)]
    items[13].visibility = elbows_visibility
    return items


def test_ohp_features_schema_and_upper_limb_gate():
    base = pose_to_ohp_features(landmarks())
    assert base is not None
    assert list(base) == list(OHP_FEATURE_NAMES)
    assert len(base) == 111
    assert len(OHP_EXTRA_NAMES) == 8
    assert len(FEATURE_NAMES) == 103
    assert OHP_SCHEMA_VERSION != "squat-103-v1"
    assert pose_to_ohp_features(landmarks(elbows_visibility=.3)) is None
    assert pose_to_ohp_features(landmarks(elbows_visibility=.3),
                                min_visibility=.25) is not None
    assert pose_to_ohp_features(None) is None
    assert pose_to_ohp_features(landmarks()[:32]) is None
    with pytest.raises(ValueError, match="min_visibility"):
        pose_to_ohp_features(landmarks(), min_visibility=1.1)


def create_source(tmp_path):
    source = tmp_path / "ohp" / "labeled"
    annotations = source / "annotations"
    annotations.mkdir(parents=True)
    splits = {"train":["100_1","101_1"],"val":["102_1"],"test":["103_1"]}
    for split, ids in splits.items():
        (annotations / f"{split}_keys.json").write_text(json.dumps(ids))
    (annotations / "error_elbows.json").write_text(json.dumps({
        "100_1": [[.35,.65],[.05,.12]],  # unsorted publisher-valid intervals
        "101_1": [], "102_1":[[.35,.65]], "103_1":[],
    }))
    (annotations / "error_knees.json").write_text(json.dumps({
        "100_1":[], "101_1":[[.35,.65]], "102_1":[], "103_1":[],
    }))
    archive_path = source / "videos.zip"
    with zipfile.ZipFile(archive_path, "w") as z:
        for key in (*splits["train"],*splits["val"],*splits["test"], "999_9"):
            z.writestr(f"videos/{key}.mp4", b"fake-payload")
    return archive_path, annotations


class Reader:
    def __init__(self):
        self.i = 0
    def isOpened(self):
        return True
    def get(self, _):
        return 10.
    def read(self):
        if self.i >= 6:
            return False, None
        self.i += 1
        return True, np.zeros((3, 3, 3), dtype=np.uint8)
    def release(self):
        pass


class CV2:
    CAP_PROP_FPS = 5
    COLOR_BGR2RGB = 6
    @staticmethod
    def VideoCapture(_):
        return Reader()
    @staticmethod
    def cvtColor(frame, _):
        return frame


class Pose:
    def __init__(self, **kwargs):
        assert kwargs["static_image_mode"]
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def process(self, _):
        return SimpleNamespace(pose_landmarks=SimpleNamespace(
            landmark=landmarks()))


MP = SimpleNamespace(solutions=SimpleNamespace(
    pose=SimpleNamespace(Pose=Pose)))


def test_unordered_elbow_labels_and_boundary_margin(tmp_path, monkeypatch):
    archive_path, _ = create_source(tmp_path)
    monkeypatch.setattr(ohp, "pose_to_ohp_features",
                        lambda _, min_visibility: {
                            name:.5 for name in OHP_FEATURE_NAMES})
    sink = io.StringIO()
    writer = csv.DictWriter(sink, fieldnames=ohp.OHP_CSV_COLUMNS)
    writer.writeheader()
    with zipfile.ZipFile(archive_path) as z:
        counters = ohp.run_clip(
            z.getinfo("videos/100_1.mp4"), z, Pose(), CV2, writer,
            video_id="100_1", split="train",
            elbow_windows=[[.35,.65],[.05,.12]], knee_windows=[],
            sample_fps=2., boundary_margin=.0, min_visibility=.5,
            max_video_bytes=1024,
        )
    rows = list(csv.DictReader(io.StringIO(sink.getvalue())))
    assert counters["pose_accepted"] == 2
    assert counters["annotated_error_elbows_frames"] == 1
    assert rows[0]["error_elbows"] == "0"
    assert rows[1]["error_elbows"] == "1"
    assert rows[0]["timestamp_s"] == "0.0"
    assert rows[1]["timestamp_s"] == "0.5"


def test_pilot_pipeline_reads_only_publisher_train_and_val(tmp_path, monkeypatch):
    archive, annotations = create_source(tmp_path)
    monkeypatch.setattr(ohp, "pose_to_ohp_features",
                        lambda _, min_visibility: {
                            name:.5 for name in OHP_FEATURE_NAMES})
    out = tmp_path / "features"
    summary = ohp.extract(
        archive, annotations, out,
        selected_splits=("train","val"), max_videos_per_split=0,
        sample_fps=2., boundary_margin=0.,
        cv2_api=CV2, mp_api=MP,
    )
    assert summary["source_test_clips_processed"] == 0
    assert set(summary["splits"]) == {"train","val"}
    assert summary["n_features"] == 111
    assert summary["splits"]["train"]["selected_videos"] == 2
    assert summary["splits"]["val"]["videos_processed"] == 1
    assert summary["splits"]["train"]["pose_accepted"] == 4
    assert summary["splits"]["val"]["pose_accepted"] == 2
    assert summary["source_extra_archive_video_ids_excluded"] == 1
    assert (out / "fitness_aqa_ohp_extraction_summary.json").is_file()
    assert not (out / "fitness_aqa_ohp_test_pose.csv").exists()
    rows = list(csv.DictReader((out / "fitness_aqa_ohp_train_pose.csv").open()))
    assert len(rows) == 4
    assert list(rows[0]) == list(ohp.OHP_CSV_COLUMNS)
    assert len({r["video_id"] for r in rows}) == 2
    assert all(r["split"] == "train" for r in rows)
    assert sum(int(r["error_elbows"]) for r in rows) == 1
    assert sum(int(r["error_knees"]) for r in rows) == 1


def test_test_source_never_accepted_for_extraction(tmp_path):
    with pytest.raises(ValueError, match="test is locked"):
        ohp.extract(tmp_path, tmp_path, tmp_path,
                    selected_splits=("test",))
    with pytest.raises(ValueError, match="test is locked"):
        ohp.extract(tmp_path, tmp_path, tmp_path,
                    selected_splits=("train","test"))


def test_negative_or_nonfinite_inputs_rejected_before_media_open(tmp_path):
    with pytest.raises(ValueError):
        ohp.extract(tmp_path, tmp_path, tmp_path, sample_fps=0)
    with pytest.raises(ValueError):
        ohp.extract(tmp_path, tmp_path, tmp_path, boundary_margin=float("nan"))
    with pytest.raises(ValueError):
        ohp.extract(tmp_path, tmp_path, tmp_path, min_visibility=1.01)
