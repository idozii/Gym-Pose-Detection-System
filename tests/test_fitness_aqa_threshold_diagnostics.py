import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from gym_pose.features import FEATURE_NAMES, SCHEMA_VERSION
from scripts.diagnose_fitness_aqa_thresholds import (
    _metrics, _clip_any_metrics, diagnose, diagnostic_report
)


def test_zero_recall_while_positive_scores_below_default_threshold():
    labels = np.asarray([0, 0, 0, 0, 1, 1])
    prob = np.asarray([.01, .10, .06, .09, .30, .40])
    vids = np.asarray(["a", "b", "c", "d", "e", "f"])
    result = diagnostic_report(labels, prob, vids, thresholds=[.2, .5])
    assert result["fixed_threshold_0_5"]["recall_error"] == 0
    assert result["fixed_threshold_0_5"]["false_negative"] == 2
    assert result["candidate_threshold_val_only_NOT_unbiased"]["threshold"] == .2
    assert result["candidate_threshold_val_only_NOT_unbiased"]["recall_error"] == 1.0
    assert result["average_precision_probability_ranking"] == pytest.approx(1.0)


def test_bad_probabilities_rejected():
    with pytest.raises(ValueError):
        diagnostic_report([0, 1], [float("nan"), .9], ["a", "b"])
    with pytest.raises(ValueError):
        diagnostic_report([0, 1], [-.1, .9], ["a", "b"])
    with pytest.raises(ValueError, match="Both"):
        diagnostic_report([0, 0], [.1, .2], ["a", "b"])


def test_video_any_acceptance_metrics_are_distinct_from_frame_metrics():
    y = np.asarray([1, 0, 0, 0])
    p = np.asarray([.4, .1, .1, .1])
    vids = np.asarray(["a", "a", "b", "b"])
    assert _metrics(y, p, .5)["false_negative"] == 1
    report = _clip_any_metrics(y, p, vids, .3)
    assert report["videos"] == 2
    assert report["true_positive"] == 1
    assert report["true_positive_videos_among_accepted_frames"] == 1


def test_diagnosis_uses_only_validation_and_trusted_models(tmp_path):
    rng = np.random.default_rng(41)
    features_dir = tmp_path / "features"
    models = tmp_path / "models"
    features_dir.mkdir()
    models.mkdir()
    training = pd.DataFrame(rng.normal(size=(50, 103)), columns=FEATURE_NAMES)
    val = pd.DataFrame(rng.normal(size=(20, 103)), columns=FEATURE_NAMES)
    val["sample_id"] = [f"clip{i // 4}_f{i}" for i in range(20)]
    val["video_id"] = [f"clip{i // 4}" for i in range(20)]
    val["split"] = "val"
    val["timestamp_s"] = [(i % 4) * .5 for i in range(20)]
    val["frame_index"] = [i % 4 for i in range(20)]
    val["source_fps"] = 30.0
    val["error_knees_forward"] = [i % 2 for i in range(20)]
    val["error_knees_inward"] = [int(i % 4 == 0) for i in range(20)]
    val["any_of_two_annotated_errors"] = val[["error_knees_forward", "error_knees_inward"]].max(axis=1)
    val.to_csv(features_dir / "fitness_aqa_val_pose.csv", index=False)
    # If implementation accidentally loads test, parsing this deliberately fails.
    (features_dir / "fitness_aqa_test_pose.csv").write_text("not,a,valid,pose,csv\n")
    for label in ("error_knees_forward", "error_knees_inward"):
        target = np.asarray([i % 2 for i in range(50)])
        model = RandomForestClassifier(n_estimators=8, random_state=42)
        model.fit(training, target)
        joblib.dump({
            "schema_version": SCHEMA_VERSION, "feature_names": list(FEATURE_NAMES),
            "target": label, "classes": [0, 1], "model": model,
        }, models / f"rf_{label}.joblib")
    output = tmp_path / "report"
    results = diagnose(features_dir, models, output)
    assert results["status"] == "validation_only_threshold_diagnostics_NOT_model_retraining"
    assert set(results["tasks"]) == {"error_knees_forward", "error_knees_inward"}
    assert (output / "threshold_grid.csv").exists()
    assert (output / "threshold_diagnostics.json").exists()
    for report in results["tasks"].values():
        assert report["fixed_threshold_0_5"]["n"] == 20
        assert report["fixed_threshold_video_any_accepted_frame"]["videos"] == 5
        assert 0 <= report["average_precision_probability_ranking"] <= 1
    assert len(list(models.glob("*.joblib"))) == 2
