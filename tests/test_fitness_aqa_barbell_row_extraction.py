"""Barbell Row publisher image/task-split fixtures; never read test pixels."""
from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gym_pose.barbell_row_features import (
    CRITICAL_JOINTS, FEATURE_NAMES, SCHEMA_VERSION, TARGETS,
    pose_to_barbell_row_features,
)
from scripts import extract_fitness_aqa_barbell_row as row


def pose_points(low=(), invalid_unused=False):
    ps = [
        SimpleNamespace(x=.03 * i, y=.015 * (i + (i % 3)),
                        z=.001 * i, visibility=.99 if i not in low else .2)
        for i in range(33)
    ]
    if invalid_unused:
        ps[0].x = float("nan")
    return ps


def test_versioned_32_feature_schema_uses_reliable_joints_only():
    good = pose_to_barbell_row_features(pose_points(invalid_unused=True))
    assert good is not None
    assert SCHEMA_VERSION == "barbell-row-image-32-v1"
    assert len(good) == 32 and list(good) == list(FEATURE_NAMES)
    assert np.isfinite(list(good.values())).all()
    assert pose_to_barbell_row_features(pose_points(low=(25,))) is None
    assert pose_to_barbell_row_features(pose_points()[:32]) is None
    degenerate = pose_points()
    for attr in ("x", "y", "z"):
        setattr(degenerate[12], attr, getattr(degenerate[11], attr))
    assert pose_to_barbell_row_features(degenerate) is None
    with pytest.raises(ValueError, match="min_visibility"):
        pose_to_barbell_row_features(pose_points(), min_visibility=float("nan"))


def source_fixture(tmp_path: Path, *, cross_task_conflict=False):
    root = tmp_path / "barbell_row" / "labeled"
    ann = root / "annotations"
    keys = {
        "train": ["201_1_0", "201_1_1", "201_1_2", "202_2_0"],
        "val": ["301_1_0", "301_1_1", "302_1_0"],
        "test": ["901_1_0", "902_1_0"],
    }
    lumbar = {split: list(ids) for split, ids in keys.items()}
    if cross_task_conflict:
        lumbar["train"][0] = "301_1_0"
        lumbar["val"][0] = "201_1_0"
    for task, splits in (("torso_angle", keys), ("lumbar", lumbar)):
        for split, ids in splits.items():
            loc = ann / "splits" / task / f"{split}_ids.json"
            loc.parent.mkdir(parents=True, exist_ok=True)
            loc.write_text(json.dumps(ids))
    allids = {item for ids in keys.values() for item in ids} | {
        item for ids in lumbar.values() for item in ids}
    for task, name in (("torso_angle", "labels_torso_angle_error.json"),
                       ("lumbar", "labels_lumbar_error.json")):
        # Both classes in each split, and different target labels.
        labels = {key: (int(key.split("_")[-1]) % 2
                        if task == "torso_angle"
                        else int(key.split("_")[-1]) % 3 == 0)
                  for key in allids}
        (ann / name).write_text(json.dumps(labels))
    zpath = root / "images.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for key in sorted(allids):
            content = b"NEVER_OPEN_TEST" if key.startswith("90") else key.encode()
            z.writestr(f"barbellrow_images_raw/{key}.jpg", content)
    return root


class FakeCv2:
    IMREAD_COLOR = 1
    COLOR_BGR2RGB = 4
    seen = []
    @classmethod
    def imdecode(cls, array, _flag):
        raw = bytes(array)
        if raw == b"NEVER_OPEN_TEST":
            raise AssertionError("SOURCE TEST IMAGE DECODED")
        cls.seen.append(raw.decode())
        return np.zeros((8, 8, 3), dtype=np.uint8)
    @staticmethod
    def cvtColor(frame, _flag):
        return frame


class FakePose:
    def __init__(self, **kwargs):
        assert kwargs["static_image_mode"] is True
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def process(self, frame):
        return SimpleNamespace(pose_landmarks=SimpleNamespace(
            landmark=pose_points()))


FakeMP = SimpleNamespace(solutions=SimpleNamespace(
    pose=SimpleNamespace(Pose=FakePose)))


def test_full_image_extraction_preserves_two_original_splits(tmp_path):
    root = source_fixture(tmp_path)
    out = tmp_path / "features"
    FakeCv2.seen = []
    report = row.extract(root / "images.zip", root / "annotations", out,
                         max_images_per_task_split=0,
                         cv2_api=FakeCv2, mp_api=FakeMP)
    assert report["source_test_images_decoded"] == 0
    assert report["media_type"] == "image"
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["cross_target_split_id_conflicts"] == 0
    assert report["splits"]["train"]["unique_images_to_decode"] == 4
    assert report["splits"]["val"]["unique_images_to_decode"] == 3
    assert len(FakeCv2.seen) == 7
    assert not any(key.startswith("90") for key in FakeCv2.seen)
    for split, count in (("train", 4), ("val", 3)):
        assert report["splits"][split]["images_decoded"] == count
        for task in TARGETS:
            stats = report["splits"][split]["tasks"][task]
            assert stats["selected_images"] == count
            assert stats["pose_accepted"] == count
            assert stats["pose_rejected"] == 0
            assert stats["positive_accepted"] + stats["negative_accepted"] == count
            outpath = out / f"fitness_aqa_barbell_row_{task}_{split}_pose.csv"
            with outpath.open(newline="") as stream:
                data = list(csv.DictReader(stream))
            assert len(data) == count
            assert list(data[0]) == list(row.COLUMNS)
            assert all(r["split"] == split and r["target"] == task for r in data)
            assert len(set(r["image_id"] for r in data)) == count
    assert (out / "fitness_aqa_barbell_row_extraction_summary.json").is_file()
    assert not list(out.glob("*_test_pose.csv"))


def test_pilot_is_label_stratified_not_test_and_does_not_mix_splits(tmp_path):
    root = source_fixture(tmp_path)
    report = row.extract(root / "images.zip", root / "annotations",
                         tmp_path / "pilot", max_images_per_task_split=2,
                         seed=42, cv2_api=FakeCv2, mp_api=FakeMP)
    assert report["max_images_per_task_split"] == 2
    for split in ("train", "val"):
        for task in TARGETS:
            target = report["splits"][split]["tasks"][task]
            assert target["selected_images"] == 2
            assert 0 <= target["positive_accepted"] <= 2
            assert target["positive_accepted"] + target["negative_accepted"] == 2


def test_test_images_and_cross_task_conflicts_blocked_before_decode(tmp_path):
    with pytest.raises(ValueError, match="TEST locked"):
        row.extract(tmp_path, tmp_path, tmp_path, selected_splits=("test",))
    root = source_fixture(tmp_path, cross_task_conflict=True)
    with pytest.raises(ValueError, match="Cross-task"):
        row.extract(root / "images.zip", root / "annotations",
                    tmp_path / "out", cv2_api=FakeCv2, mp_api=FakeMP)


def test_failing_images_reported_without_inserting_fabricated_negative_rows(
        tmp_path, monkeypatch):
    root = source_fixture(tmp_path)
    class BrokenCv(FakeCv2):
        @classmethod
        def imdecode(cls, array, flag):
            if bytes(array) == b"201_1_0":
                return None
            return super().imdecode(array, flag)
    out = tmp_path / "broken"
    result = row.extract(root / "images.zip", root / "annotations", out,
                         max_images_per_task_split=0,
                         cv2_api=BrokenCv, mp_api=FakeMP)
    assert result["splits"]["train"]["image_decode_failures"] == 1
    for task in TARGETS:
        stats = result["splits"]["train"]["tasks"][task]
        assert stats["pose_accepted"] == 3
        assert stats["pose_rejected"] == 1
    with (out / "fitness_aqa_barbell_row_torso_angle_train_pose.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 3


def test_only_required_zip_id_is_opened_and_conflict_checks_are_conservative(
        tmp_path):
    root = source_fixture(tmp_path)
    with zipfile.ZipFile(root / "images.zip") as z:
        entries = row._index_media(z, {"201_1_0", "301_1_1"})
    assert len(entries) == 2
    assert "901_1_0" not in entries
    mapping = {
        "torso_angle": {"200_1_0": "train", "200_1_1": "val"},
        "lumbar": {"500_1_0": "train", "501_1_0": "val"},
    }
    x = row._prefix_overlap(mapping)
    assert x["torso_angle"]["candidate_prefix_overlap_train_val"] == 1
    assert x["lumbar"]["candidate_prefix_overlap_train_val"] == 0


def test_invalid_options_rejected_before_video_or_image_open(tmp_path):
    for kwargs in (
        {"max_images_per_task_split": -1},
        {"min_visibility": float("nan")},
        {"min_visibility": 1.4},
        {"max_pixels": 0},
        {"max_image_mb": 0},
        {"selected_splits": ("train", "test")},
    ):
        with pytest.raises(ValueError):
            row.extract(tmp_path, tmp_path, tmp_path, **kwargs)
