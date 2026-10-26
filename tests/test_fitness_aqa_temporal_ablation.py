"""Synthetic tests only: no real dataset metrics or test-set selection."""
import json

import numpy as np
import pandas as pd
import pytest

from gym_pose.features import FEATURE_NAMES
from scripts.compare_fitness_aqa_temporal import (
    TEMPORAL_NAMES, choose_threshold, features, grouped_calibration_split, run,
)


def synthetic_frame_table(*, clips=16, frames=9, split="train"):
    rng = np.random.default_rng(17 if split == "train" else 71)
    rows = []
    for clip in range(clips):
        for frame in range(frames):
            base = dict(zip(FEATURE_NAMES, rng.normal(size=103)))
            base.update({
                "video_id": f"{split}_clip{clip}", "sample_id": f"{split}_clip{clip}_f{frame}",
                "split": split, "timestamp_s": .5 * frame, "frame_index": frame,
                "source_fps": 30.0,
                "error_knees_forward": int(clip % 3 == 0 and frame % 2 == 0),
                "error_knees_inward": int(clip % 4 == 0 and frame < 6),
            })
            base["any_of_two_annotated_errors"] = max(
                base["error_knees_forward"], base["error_knees_inward"])
            rows.append(base)
    return pd.DataFrame(rows)


def test_temporal_features_are_causal_and_not_cross_video():
    df = synthetic_frame_table(clips=2, frames=4)
    base = features(df, temporal=False)
    temporal = features(df, temporal=True)
    assert base.shape[1] == 103
    assert temporal.shape[1] == 103 + len(TEMPORAL_NAMES) == 119
    assert temporal.iloc[0]["knee_angle_mean_delta1"] == 0
    assert temporal.iloc[4]["knee_angle_mean_delta1"] == 0  # new video
    original = df.copy()
    original.loc[3, "knee_angle_mean"] += 500  # future of frame 1
    new = features(original, temporal=True)
    assert new.loc[1, "knee_angle_mean_causal_mean3"] == pytest.approx(
        temporal.loc[1, "knee_angle_mean_causal_mean3"])


def test_temporal_segment_resets_after_gap():
    df = synthetic_frame_table(clips=1, frames=3)
    df.loc[2, "timestamp_s"] = 2.0
    x = features(df, temporal=True)
    assert x.loc[2, "hip_angle_mean_delta1"] == 0
    assert x.loc[2, "hip_angle_mean_causal_mean3"] == pytest.approx(
        x.loc[2, "hip_angle_mean"])


def test_group_calibration_never_mixes_videos():
    df = synthetic_frame_table()
    fit, cal = grouped_calibration_split(df, seed=42)
    assert set(df.iloc[fit].video_id).isdisjoint(set(df.iloc[cal].video_id))
    assert all(set(df.iloc[indices][label]) == {0, 1}
               for indices in (fit, cal)
               for label in ("error_knees_forward", "error_knees_inward"))


def test_threshold_chosen_without_validation_input():
    y = np.asarray([0, 0, 0, 1, 1])
    probabilities = np.asarray([.01, .05, .11, .21, .29])
    choice = choose_threshold(y, probabilities)
    assert choice["threshold"] < .5
    assert choice["recall_error"] == 1


def test_full_experiment_train_group_calibrates_without_test(tmp_path):
    train = synthetic_frame_table()
    val = synthetic_frame_table(clips=8, split="val")
    data = tmp_path / "features"
    data.mkdir()
    train.to_csv(data / "fitness_aqa_train_pose.csv", index=False)
    val.to_csv(data / "fitness_aqa_val_pose.csv", index=False)
    (data / "fitness_aqa_test_pose.csv").write_text("INVALID,SHOULD,NOT,BE,OPENED\n")
    output = tmp_path / "results"
    report = run(data, output, seed=42, n_estimators=12)
    assert report["status"] == "train_group_calibrated_validation_evaluation_ONLY"
    assert report["train_videos"] == 16
    assert report["validation_videos"] == 8
    assert set(report["variants"]) == {"static", "causal_temporal"}
    assert report["variants"]["static"]["features_count"] == 103
    assert report["variants"]["causal_temporal"]["features_count"] == 119
    for variant in report["variants"].values():
        for task in variant["tasks"].values():
            assert task["validation_fixed_0_5"]["n"] == len(val)
            assert 0 <= task["train_calibrated_threshold"] <= 1
            assert 0 <= task["validation_average_precision"] <= 1
    assert len(list(output.glob("*.joblib"))) == 4
    assert json.loads((output / "comparison.json").read_text())["seed"] == 42


def test_reject_cross_split_video_leakage(tmp_path):
    df = synthetic_frame_table()
    val = synthetic_frame_table(split="val", clips=8)
    val.loc[0, "video_id"] = df.loc[0, "video_id"]
    directory = tmp_path / "features"
    directory.mkdir()
    df.to_csv(directory / "fitness_aqa_train_pose.csv", index=False)
    val.to_csv(directory / "fitness_aqa_val_pose.csv", index=False)
    with pytest.raises(ValueError, match="overlap"):
        run(directory, tmp_path / "output", n_estimators=2)
