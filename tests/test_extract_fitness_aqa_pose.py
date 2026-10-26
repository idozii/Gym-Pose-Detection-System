import csv
import io
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from gym_pose.features import FEATURE_NAMES
from scripts import extract_fitness_aqa_pose as prep


def test_interval_labels_and_ambiguous_boundary():
    assert prep.labels_at_time(0.0, [[1.0, 2.0]], [], margin=.1) == (0, 0)
    assert prep.labels_at_time(1.5, [[1.0, 2.0]], [], margin=.1) == (1, 0)
    assert prep.labels_at_time(3.5, [], [[3.0, 4.0]], margin=.1) == (0, 1)
    assert prep.labels_at_time(1.03, [[1.0, 2.0]], [], margin=.1) is None
    assert prep.labels_at_time(2.04, [[1.0, 2.0]], [], margin=.1) is None


def test_archive_member_selection_ignores_extra_videos(tmp_path):
    path = tmp_path / "videos.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("videos/123_1.mp4", b"a")
        z.writestr("videos/123_2.mp4", b"b")
        z.writestr("videos/999_9.mp4", b"extra")
    with zipfile.ZipFile(path) as z:
        selected = prep.video_members(z, {"123_1", "123_2"})
        assert sorted(selected) == ["123_1", "123_2"]
        with pytest.raises(ValueError, match="Missing"):
            prep.video_members(z, {"321_1"})


def test_one_clip_processing_preserves_timestamp_and_two_error_targets(tmp_path, monkeypatch):
    archive_path = tmp_path / "clips.zip"
    with zipfile.ZipFile(archive_path, "w") as z:
        z.writestr("videos/123_1.mp4", b"fake clip payload")

    class VideoReader:
        def __init__(self):
            self.index = 0
        def isOpened(self):
            return True
        def get(self, _):
            return 10.0
        def read(self):
            if self.index >= 6:
                return False, None
            self.index += 1
            return True, np.zeros((3, 3, 3), dtype=np.uint8)
        def release(self):
            pass

    class CV2:
        CAP_PROP_FPS = 5
        COLOR_BGR2RGB = 6
        @staticmethod
        def VideoCapture(_):
            return VideoReader()
        @staticmethod
        def cvtColor(frame, _):
            return frame

    class Pose:
        @staticmethod
        def process(_):
            return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=object()))

    monkeypatch.setattr(prep, "pose_to_features", lambda _, min_visibility: {
        key: 0.5 for key in FEATURE_NAMES
    })
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=prep.CSV_COLUMNS)
    writer.writeheader()
    with zipfile.ZipFile(archive_path) as z:
        stats = prep._run_clip(
            z.getinfo("videos/123_1.mp4"), z, Pose(), CV2, writer,
            key="123_1", split="train", forward=[[.4, .8]], inward=[],
            sample_fps=2.0, margin=0.0, visibility=.5, max_video_bytes=1024
        )
    rows = list(csv.DictReader(io.StringIO(out.getvalue())))
    assert len(rows) == 2
    assert rows[0]["timestamp_s"] == "0.0"
    assert rows[1]["timestamp_s"] == "0.5"
    assert [r["error_knees_forward"] for r in rows] == ["0", "1"]
    assert [r["error_knees_inward"] for r in rows] == ["0", "0"]
    assert stats["pose_accepted"] == 2
    assert not (tmp_path / "clip.mp4").exists()


def test_validation_fails_on_invalid_sampling(tmp_path):
    with pytest.raises(ValueError):
        prep.extract(tmp_path, tmp_path, tmp_path, sample_fps=0)
