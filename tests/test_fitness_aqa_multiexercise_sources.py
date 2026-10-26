"""Synthetic publisher-compatible fixtures: no external ZIP download required."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from gym_pose.fitness_aqa_sources import (
    inspect_barbell_row, inspect_ohp, inspect_zip, load_splits,
    validate_binary_labels, validate_intervals, OHP_KEY,
)
from scripts.inspect_fitness_aqa_multi import inspect


def _json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_ohp(root: Path):
    base = root / "ohp" / "labeled"
    ids = {"train": ["101_1", "101_2"], "val": ["102_1"], "test": ["103_1"]}
    for split, items in ids.items():
        _json(base / "annotations" / f"{split}_keys.json", items)
    _json(base / "annotations" / "error_elbows.json",
          {"101_1": [[.1, .8]], "101_2": [], "102_1": [], "103_1": []})
    _json(base / "annotations" / "error_knees.json",
          {"101_1": [], "101_2": [[.2, .9]], "102_1": [], "103_1": []})
    with zipfile.ZipFile(base / "videos.zip", "w") as z:
        for key in ("101_1", "101_2", "102_1", "103_1", "999_9"):
            z.writestr(f"publisher/{key}.mp4", b"not actual video")
    return base


def make_row(root: Path, *, conflict: bool = False):
    base = root / "barbell_row" / "labeled"
    torso = {"train": ["201_1_0", "201_1_1"], "val": ["202_1_0"], "test": ["203_1_0"]}
    lumbar = {"train": ["201_1_0", "201_1_1"], "val": ["202_1_0"], "test": ["203_1_0"]}
    if conflict:
        lumbar = {"train": ["201_1_0", "202_1_0"],
                  "val": ["201_1_1"], "test": ["203_1_0"]}
    for task, splits in (("torso_angle", torso), ("lumbar", lumbar)):
        for name, keys in splits.items():
            _json(base / "annotations" / "splits" / task / f"{name}_ids.json", keys)
    _json(base / "annotations" / "labels_torso_angle_error.json",
          {"201_1_0": 0, "201_1_1": 1, "202_1_0": 0, "203_1_0": 1})
    _json(base / "annotations" / "labels_lumbar_error.json",
          {"201_1_0": 1, "201_1_1": 0, "202_1_0": 1, "203_1_0": 0})
    with zipfile.ZipFile(base / "images.zip", "w") as z:
        for name in ("201_1_0", "201_1_1", "202_1_0", "203_1_0"):
            z.writestr(f"barbellrow_images_raw/{name}.jpg", b"not actual image")
    return base


def test_ohp_source_manifest_with_extra_video(tmp_path):
    base = make_ohp(tmp_path)
    manifest = inspect_ohp(base)
    assert manifest["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert manifest["assigned_unique_video_ids"] == 4
    assert manifest["targets"]["error_elbows"]["videos_with_annotated_error"] == 1
    assert manifest["archive"]["extra_keys_count"] == 1
    assert manifest["ready_for_pose_extraction"]
    assert not (tmp_path / "publisher").exists()


def test_ohp_metadata_only_does_not_require_archive(tmp_path):
    base = make_ohp(tmp_path)
    (base / "videos.zip").unlink()
    report = inspect_ohp(base, skip_archive=True)
    assert report["archive"]["archive_checked"] is False
    assert report["ready_for_pose_extraction"] is False
    assert report["targets"]["error_knees"]["annotated_intervals"] == 1


def test_invalid_ohp_intervals_and_duplicate_split_rejected(tmp_path):
    base = make_ohp(tmp_path)
    _json(base / "annotations" / "error_elbows.json",
          {"101_1": [[2.0, 1.0]], "101_2": [], "102_1": [], "103_1": []})
    with pytest.raises(ValueError, match="invalid annotation interval"):
        inspect_ohp(base)
    _json(base / "annotations" / "error_elbows.json",
          {"101_1": [], "101_2": [], "102_1": [], "103_1": []})
    _json(base / "annotations" / "val_keys.json", ["101_2"])
    with pytest.raises(ValueError, match="Data leakage"):
        inspect_ohp(base)


def test_label_completeness_and_binary_type_enforced():
    with pytest.raises(ValueError, match="missing label"):
        validate_binary_labels({"201_1_0": 0}, {"201_1_0", "202_1_0"}, task="torso")
    with pytest.raises(ValueError, match="integer 0 or 1"):
        validate_binary_labels({"201_1_0": True}, {"201_1_0"}, task="lumbar")
    with pytest.raises(ValueError, match="missing annotation"):
        validate_intervals({}, {"101_1"}, task="elbows")


def test_row_preserves_two_separate_label_splits(tmp_path):
    base = make_row(tmp_path)
    manifest = inspect_barbell_row(base)
    assert manifest["ready_for_task_separate_image_extraction"]
    assert manifest["unique_images_required_across_tasks"] == 4
    assert manifest["tasks"]["torso_angle"]["targets"]["torso_angle"]["positive_images"] == 2
    assert manifest["tasks"]["lumbar"]["targets"]["lumbar"]["positive_images"] == 2
    assert manifest["cross_task_split_audit"]["images_with_different_splits_between_tasks"] == 0
    assert manifest["archive"]["missing_keys_count"] == 0
    assert manifest["ready_for_unified_multitask_training"] is True  # no ID or candidate-prefix conflict in this fixture


def test_cross_task_image_split_conflicts_detected_without_rejection(tmp_path):
    base = make_row(tmp_path, conflict=True)
    report = inspect_barbell_row(base)
    audit = report["cross_task_split_audit"]
    assert report["ready_for_task_separate_image_extraction"] is True
    assert audit["images_with_different_splits_between_tasks"] == 2
    assert audit["safe_to_naively_merge_into_joint_multitask_split"] is False
    assert report["ready_for_unified_multitask_training"] is False


def test_row_zip_missing_and_duplicate_image_ids_reported(tmp_path):
    base = make_row(tmp_path)
    with zipfile.ZipFile(base / "images.zip", "w") as z:
        z.writestr("x/201_1_0.png", b"a")
        z.writestr("y/201_1_0.jpg", b"b")
    out = inspect_barbell_row(base)
    assert out["archive"]["missing_keys_count"] == 3
    assert out["archive"]["duplicates_count"] == 1
    assert out["ready_for_task_separate_image_extraction"] is False


def test_cli_combines_checked_exercises_and_no_media_extraction(tmp_path):
    make_ohp(tmp_path)
    make_row(tmp_path)
    report = inspect(tmp_path, exercises=("ohp", "barbell_row"))
    assert report["milestone"] == "12"
    assert report["all_requested_archives_ready"] is True
    assert set(report["results"]) == {"ohp", "barbell_row"}
    assert report["results"]["barbell_row"]["annotation_semantics"] == "per_image_binary_error_label"


def test_metadata_only_across_exercises_without_archives(tmp_path):
    ohp = make_ohp(tmp_path)
    row = make_row(tmp_path)
    (ohp / "videos.zip").unlink()
    (row / "images.zip").unlink()
    report = inspect(tmp_path, exercises=("ohp", "barbell_row"), skip_archive=True)
    assert report["all_requested_archives_ready"] is False
    assert all(not v["archive"]["archive_checked"] for v in report["results"].values())


def test_invalid_row_source_key_structure(tmp_path):
    base = make_row(tmp_path)
    _json(base / "annotations" / "splits" / "lumbar" / "train_ids.json", ["201_1_0.jpg"])
    with pytest.raises(ValueError, match="valid publisher"):
        inspect_barbell_row(base)


def test_ohp_publisher_unsorted_intervals_are_valid_and_not_modified(tmp_path):
    base = make_ohp(tmp_path)
    # Real OHP elbow source 79916_1 has windows with start times 2.88 and 0.60.
    original = [[2.88, 3.6], [.6, 1.33]]
    overlapping = [[0., .53], [.52, 1.87]]
    labels = {"101_1": original, "101_2": overlapping,
              "102_1": [], "103_1": []}
    _json(base / "annotations" / "error_elbows.json", labels)
    result = inspect_ohp(base)
    elbow = result["targets"]["error_elbows"]
    assert elbow["videos_with_annotated_error"] == 2
    assert elbow["annotated_intervals"] == 4
    assert elbow["videos_with_out_of_order_intervals"] == 1
    assert elbow["out_of_order_video_examples"] == ["101_1"]
    assert elbow["videos_with_overlapping_intervals"] == 1
    assert elbow["overlapping_video_examples"] == ["101_2"]
    assert labels["101_1"] == original
    assert json.loads((base / "annotations" / "error_elbows.json").read_text())["101_1"] == original


@pytest.mark.parametrize("bad", [
    [[-1., 1.]], [[1., 1.]], [[2., 1.]], [[float("inf"), 3.]],
    [[float("nan"), 3.]], [[0., "bad"]], [[0.]], [[0., 1., 2.]],
])
def test_ohp_invalid_intervals_still_rejected(bad):
    with pytest.raises(ValueError, match="invalid annotation interval"):
        validate_intervals({"101_1": bad}, {"101_1"}, task="error_elbows")
