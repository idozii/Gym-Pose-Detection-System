import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.audit_fitness_aqa_video_quality import (
    annotation_stratum, stratified_video_sample, summarize_rows, audit
)


def test_stratified_sampling_is_deterministic_and_balanced():
    forward = {str(k): ([[.1, .9]] if k % 3 == 1 else [])
               for k in range(30)}
    inward = {str(k): ([[.1, .9]] if k % 3 == 0 else [])
              for k in range(30)}
    keys = [str(k) for k in range(30)]
    a = stratified_video_sample(keys, forward, inward, count=9, seed=42)
    b = stratified_video_sample(keys, forward, inward, count=9, seed=42)
    assert a == b
    assert len({k for k, _ in a}) == 9
    assert sum(s == "inward_annotated_video" for _, s in a) >= 3
    assert sum(s == "forward_only_annotated_video" for _, s in a) >= 3
    assert sum(s == "neither_of_two_annotated_errors" for _, s in a) >= 3


def test_summary_counts_rejected_and_label_bias():
    rows = [
        {"video_id":"a", "video_annotation_stratum":"inward_annotated_video",
         "rejection_reason":"accepted", "low_visibility_joints":[],
         "error_knees_forward":0,"error_knees_inward":1},
        {"video_id":"a", "video_annotation_stratum":"inward_annotated_video",
         "rejection_reason":"critical_joint_low_visibility",
         "low_visibility_joints":["left_knee"],"error_knees_forward":0,
         "error_knees_inward":1},
        {"video_id":"b", "video_annotation_stratum":"neither_of_two_annotated_errors",
         "rejection_reason":"no_pose_detected",
         "low_visibility_joints":[],"error_knees_forward":0,"error_knees_inward":0},
    ]
    stats = summarize_rows(rows, 2, [], [("a", "inward_annotated_video"),
                                         ("b", "neither_of_two_annotated_errors")])
    assert stats["eligible_frames"] == 3
    assert stats["accepted_frames"] == 1
    assert stats["low_visibility_joint_counts"]["left_knee"] == 1
    assert stats["by_error_annotation_at_timestamp"]["error_knees_inward"]["1"]["acceptance_fraction"] == .5


def fake_joints(visibility=1.0):
    return [SimpleNamespace(x=.02 * i, y=.03 * i, z=.0,
                            visibility=(visibility if i == 25 else 1.0)) for i in range(33)]


def test_end_to_end_samples_train_val_not_test(tmp_path):
    archive_path = tmp_path / "videos.zip"
    train, val = ["101_1", "102_1", "103_1"], ["201_1", "202_1", "203_1"]
    test = ["999_1"]
    all_ids = train + val
    with zipfile.ZipFile(archive_path, "w") as z:
        for key in all_ids + test:
            z.writestr(f"videos/{key}.mp4", b"fake")
    anns = tmp_path / "annotations"
    anns.mkdir()
    for name, keys in (("train", train), ("val", val)):
        (anns / f"{name}_keys.json").write_text(json.dumps(keys))
    (anns / "error_knees_forward.json").write_text(json.dumps({
        key: ([[.25, .8]] if key.endswith("2_1") else []) for key in all_ids
    }))
    (anns / "error_knees_inward.json").write_text(json.dumps({
        key: ([[.25, .8]] if key.endswith("3_1") else []) for key in all_ids
    }))

    class Reader:
        def __init__(self):
            self.i = 0
        def isOpened(self): return True
        def get(self, prop): return 10.0
        def read(self):
            if self.i >= 6: return False, None
            self.i += 1
            return True, np.zeros((3, 3, 3), dtype=np.uint8)
        def release(self): pass

    class CV2:
        CAP_PROP_FPS = 5
        COLOR_BGR2RGB = 6
        @staticmethod
        def VideoCapture(_): return Reader()
        @staticmethod
        def cvtColor(frame, _): return frame

    class PoseEstimator:
        def __init__(self, **kwargs): self.n = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def process(self, image):
            self.n += 1
            if self.n % 3 == 1:
                return SimpleNamespace(pose_landmarks=None)
            elif self.n % 3 == 2:
                return SimpleNamespace(pose_landmarks=SimpleNamespace(
                    landmark=fake_joints(.3)))
            return SimpleNamespace(pose_landmarks=SimpleNamespace(
                landmark=fake_joints(1.0)))

    mp = SimpleNamespace(solutions=SimpleNamespace(
        pose=SimpleNamespace(Pose=PoseEstimator)))
    result = audit(archive_path, anns, tmp_path / "output",
                   videos_per_split=3, sample_fps=2, boundary_margin=0.0,
                   cv2_api=CV2, mp_api=mp)
    assert set(result["splits"]) == {"train", "val"}
    assert sum(v["eligible_frames"] for v in result["splits"].values()) == 12
    assert all(v["selected_videos"] == 3 for v in result["splits"].values())
    assert result["splits"]["train"]["rejection_reason_counts"]["no_pose_detected"] > 0
    assert result["splits"]["train"]["rejection_reason_counts"]["critical_joint_low_visibility"] > 0
    assert (tmp_path / "output" / "pose_quality_report.json").is_file()
    assert (tmp_path / "output" / "quality_val_samples.csv").is_file()
    assert not (tmp_path / "output" / "quality_test_samples.csv").exists()


def test_reject_invalid_pilot_size(tmp_path):
    with pytest.raises(ValueError):
        audit(tmp_path, tmp_path, tmp_path, videos_per_split=0)
