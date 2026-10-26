import json

import pandas as pd
import pytest

from scripts.diagnose_fitness_aqa_joint_gates import (
    ANKLES, KNEES, counterfactual_accept, diagnose, summarize_split,
)


def test_joint_selective_counterfactual_requires_only_relaxed_joints():
    assert counterfactual_accept("accepted", "", 0.8, relaxed_joints=ANKLES)
    assert counterfactual_accept("critical_joint_low_visibility", "left_ankle",
                                 .38, relaxed_joints=ANKLES)
    assert not counterfactual_accept("critical_joint_low_visibility", "left_knee",
                                     .38, relaxed_joints=ANKLES)
    assert counterfactual_accept("critical_joint_low_visibility", "left_knee",
                                 .38, relaxed_joints=KNEES)
    assert not counterfactual_accept("critical_joint_low_visibility",
                                     "left_knee|left_ankle", .38, relaxed_joints=KNEES)
    assert not counterfactual_accept("critical_joint_low_visibility", "left_ankle",
                                     .25, relaxed_joints=ANKLES)
    assert not counterfactual_accept("no_pose_detected", "", None, relaxed_joints=ANKLES)


def _sample():
    return pd.DataFrame([
        {"video_id":"a", "sample_id":"a_0", "rejection_reason":"accepted",
         "low_visibility_joints":"", "minimum_critical_joint_visibility":.91,
         "error_knees_forward":0, "error_knees_inward":0},
        {"video_id":"a", "sample_id":"a_1", "rejection_reason":"critical_joint_low_visibility",
         "low_visibility_joints":"left_ankle", "minimum_critical_joint_visibility":.4,
         "error_knees_forward":0, "error_knees_inward":1},
        {"video_id":"b", "sample_id":"b_1", "rejection_reason":"critical_joint_low_visibility",
         "low_visibility_joints":"right_knee", "minimum_critical_joint_visibility":.4,
         "error_knees_forward":1, "error_knees_inward":1},
        {"video_id":"b", "sample_id":"b_2", "rejection_reason":"critical_joint_low_visibility",
         "low_visibility_joints":"left_knee|left_ankle", "minimum_critical_joint_visibility":.4,
         "error_knees_forward":1, "error_knees_inward":0},
        {"video_id":"b", "sample_id":"b_3", "rejection_reason":"critical_joint_low_visibility",
         "low_visibility_joints":"left_ankle", "minimum_critical_joint_visibility":.2,
         "error_knees_forward":0, "error_knees_inward":0},
        {"video_id":"c", "sample_id":"c_1", "rejection_reason":"no_pose_detected",
         "low_visibility_joints":"", "minimum_critical_joint_visibility":None,
         "error_knees_forward":0, "error_knees_inward":0},
    ])


def test_count_gate_recovery_and_positive_label_specificity():
    result = summarize_split(_sample(), expected_050=1, expected_035=4)
    assert result["all_050"]["accepted_frames"] == 1
    assert result["ankles_035_other_050"]["accepted_frames"] == 2
    assert result["knees_035_other_050"]["accepted_frames"] == 2
    assert result["knees_ankles_035_other_050"]["accepted_frames"] == 4
    assert result["all_035"]["accepted_frames"] == 4
    inward = result["ankles_035_other_050"]["error_targets"]["error_knees_inward"]
    assert inward["new_positive_vs_050"] == 1
    assert inward["new_negative_vs_050"] == 0
    assert result["ankles_035_other_050"]["videos_with_accepted_frames"] == 1


def test_zero_record_baseline_mismatch_blocks_false_qa_claim():
    with pytest.raises(ValueError, match="Baseline"):
        summarize_split(_sample(), expected_050=2, expected_035=4)


def test_cli_function_reuses_existing_csv_no_video_or_test_read(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    sample = _sample()
    raw = sample.drop(columns=["sample_id"]).copy()
    raw["frame_index"] = [0, 1, 1, 2, 3, 1]
    summary = {"status":"pose_quality_DIAGNOSTIC_only_no_training_no_test",
               "baseline_min_visibility":0.5,"splits":{}}
    for split in ("train", "val"):
        raw.to_csv(path / f"quality_{split}_samples.csv", index=False)
        summary["splits"][split] = {
            "selected_videos": 3, "eligible_frames": len(raw),
            "accepted_frames": 1,
            "cutoff_accepted_counts_same_eligible_frames":{"0.35":{"accepted":4}},
        }
    (path / "pose_quality_report.json").write_text(json.dumps(summary))
    (path / "quality_test_samples.csv").write_text("corrupt,SHOULD_NOT_BE_OPENED\n")
    output = tmp_path / "report" / "counterfactual.json"
    result = diagnose(path, output)
    assert result["splits"]["train"]["gates"]["ankles_035_other_050"]["accepted_frames"] == 2
    assert result["splits"]["val"]["gates"]["all_035"]["accepted_frames"] == 4
    assert output.is_file()


def test_no_invalid_joint_labels():
    with pytest.raises(ValueError, match="unknown"):
        counterfactual_accept("critical_joint_low_visibility", "mystery",
                              .45, relaxed_joints=ANKLES)
