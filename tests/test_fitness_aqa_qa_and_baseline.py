import json

import numpy as np
import pandas as pd
import pytest

from gym_pose.features import FEATURE_NAMES
from scripts.audit_fitness_aqa_pose import audit
from scripts.benchmark_fitness_aqa_errors import benchmark


def _sample_data(tmp_path, *, clips_per_split=8, frames_per_clip=8):
    folder = tmp_path / "features"
    folder.mkdir()
    rng = np.random.default_rng(12)
    summary = {
        "split_type": "publisher_video_id_split_NOT_subject_disjoint",
        "splits": {},
    }
    for split in ("train", "val", "test"):
        records = []
        for vid in range(clips_per_split):
            video = f"{split}_{vid}"  # synthetic IDs, NOT Fitness-AQA original video IDs
            for frame in range(frames_per_clip):
                forward = int(vid < clips_per_split // 2 and frame < frames_per_clip // 2)
                inward = int(vid % 2 == 0 and frame < frames_per_clip // 2)
                row = {col: float(v) for col, v in zip(
                    FEATURE_NAMES, rng.normal(size=len(FEATURE_NAMES)))}
                row.update({
                    "sample_id": f"{video}_f{frame:07d}",
                    "video_id": video, "split": split,
                    "timestamp_s": frame / 2.0, "frame_index": frame,
                    "source_fps": 30.0,
                    "error_knees_forward": forward, "error_knees_inward": inward,
                    "any_of_two_annotated_errors": max(forward, inward),
                })
                records.append(row)
        pd.DataFrame(records).to_csv(folder / f"fitness_aqa_{split}_pose.csv", index=False)
        n = len(records)
        summary["splits"][split] = {
            "selected_videos": clips_per_split,
            "frames_sampled": n + 3, "boundary_rejected": 3,
            "pose_accepted": n, "pose_rejected": 0,
        }
    (folder / "fitness_aqa_extraction_summary.json").write_text(json.dumps(summary))
    return folder


def test_qa_counts_coverage_and_positives(tmp_path):
    folder = _sample_data(tmp_path)
    result = audit(folder)
    assert result["splits"]["train"]["pose_coverage_eligible"] == 1.0
    assert result["splits"]["train"]["targets"]["error_knees_forward"]["positive_frames"] == 16
    assert result["splits"]["val"]["videos_with_accepted_frames"] == 8


def test_qa_catches_cross_split_video_leakage(tmp_path):
    folder = _sample_data(tmp_path)
    val = folder / "fitness_aqa_val_pose.csv"
    df = pd.read_csv(val)
    df.loc[0, "video_id"] = "train_0"
    df.to_csv(val, index=False)
    with pytest.raises(ValueError, match="Split leakage"):
        audit(folder)


def test_qa_rejects_mismatched_extraction_summary(tmp_path):
    folder = _sample_data(tmp_path)
    p = folder / "fitness_aqa_extraction_summary.json"
    summary = json.loads(p.read_text())
    summary["splits"]["train"]["pose_accepted"] += 1
    p.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="row count"):
        audit(folder)


def test_baseline_refuses_tiny_pilot(tmp_path):
    folder = _sample_data(tmp_path)
    with pytest.raises(ValueError, match="Need >="):
        benchmark(folder, tmp_path / "model", min_positive_frames=20, min_positive_videos=5,
                  n_estimators=10)
    assert not (tmp_path / "model").exists()


def test_baseline_uses_train_validation_only(tmp_path):
    folder = _sample_data(tmp_path, clips_per_split=12, frames_per_clip=12)
    # The test CSV deliberately contains invalid values: the trainer must NEVER open it.
    (folder / "fitness_aqa_test_pose.csv").write_text("invalid,test,content\n")
    report = benchmark(folder, tmp_path / "model", min_positive_frames=10,
                       min_positive_videos=4, n_estimators=20)
    assert report["status"] == "validation_only_NO_test_used"
    assert set(report["tasks"]) == {"error_knees_forward", "error_knees_inward"}
    for metrics in report["tasks"].values():
        assert metrics["videos"] == 12
        assert 0 <= metrics["macro_f1_frame"] <= 1
    assert (tmp_path / "model" / "validation_report.json").is_file()
    assert (tmp_path / "model" / "rf_error_knees_forward.joblib").is_file()
