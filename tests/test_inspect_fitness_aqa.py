import json
import zipfile

import pytest

from scripts.inspect_fitness_aqa import inspect


def _fixtures(tmp_path):
    anns = tmp_path / "annotations"
    anns.mkdir()
    splits = {"train": ["100_1", "101_2"], "val": ["102_3"], "test": ["103_4"]}
    for name, keys in splits.items():
        (anns / f"{name}_keys.json").write_text(json.dumps(keys))
    annotations = {
        "100_1": [[0.5, 1.2]], "101_2": [],
        "102_3": [[0.1, 0.7]], "103_4": [],
    }
    for name in ("error_knees_forward", "error_knees_inward"):
        (anns / f"{name}.json").write_text(json.dumps(annotations))
    zip_path = tmp_path / "videos.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for key in annotations:
            archive.writestr(f"videos/{key}.mp4", b"mock-not-video")
    return zip_path, anns


def test_real_zip_is_not_extracted(tmp_path):
    archive, anns = _fixtures(tmp_path)
    result = inspect(archive, anns)
    assert result["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert result["matched_video_keys"] == 4
    assert result["missing_video_keys_count"] == 0
    assert result["annotation_video_counts"]["knees_forward"] == 2
    assert not (tmp_path / "videos").exists()


def test_split_leakage_fails(tmp_path):
    archive, anns = _fixtures(tmp_path)
    (anns / "test_keys.json").write_text(json.dumps(["100_1"]))
    with pytest.raises(ValueError, match="Data leakage"):
        inspect(archive, anns)


def test_annotation_missing_keys_fails(tmp_path):
    archive, anns = _fixtures(tmp_path)
    (anns / "error_knees_inward.json").write_text(json.dumps({"100_1": []}))
    with pytest.raises(ValueError, match="Missing"):
        inspect(archive, anns)


def test_archive_missing_video_is_reported(tmp_path):
    archive, anns = _fixtures(tmp_path)
    archive.unlink()
    with zipfile.ZipFile(archive, "w") as sink:
        sink.writestr("videos/100_1.mp4", b"mock")
    result = inspect(archive, anns)
    assert result["missing_video_keys_count"] == 3
