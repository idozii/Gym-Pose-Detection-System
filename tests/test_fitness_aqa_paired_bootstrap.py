import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from gym_pose.features import FEATURE_NAMES, SCHEMA_VERSION
from scripts.compare_fitness_aqa_temporal import TEMPORAL_SCHEMA, features
from scripts.fitness_aqa_paired_bootstrap import compare, paired_cluster_bootstrap


def test_paired_bootstrap_is_deterministic_and_clustered():
    y = np.asarray([1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    vids = np.repeat(["clip_a", "clip_b", "clip_c"], 4)
    static = np.asarray([.2, .5, .3, .4, .5, .1, .4, .3, .6, .3, .3, .1])
    temporal = np.asarray([.8, .1, .2, .1, .2, .9, .2, .1, .2, .85, .2, .1])
    a = paired_cluster_bootstrap(y, static, temporal, vids,
                                 static_threshold=.4, temporal_threshold=.4,
                                 repeats=100, seed=42)
    b = paired_cluster_bootstrap(y, static, temporal, vids,
                                 static_threshold=.4, temporal_threshold=.4,
                                 repeats=100, seed=42)
    assert a == b
    assert a["videos"] == 3
    assert a["frames"] == 12
    assert a["paired_differences"]["average_precision"]["delta_temporal_minus_static"] > 0
    assert a["paired_differences"]["macro_f1"]["delta_temporal_minus_static"] > 0
    assert a["valid_bootstrap_samples"] >= 50


def test_invalid_inputs_fail_safely():
    with pytest.raises(ValueError):
        paired_cluster_bootstrap([0, 1], [.3, float("nan")], [.4, .8], ["a", "b"],
                                 static_threshold=.5, temporal_threshold=.5)
    with pytest.raises(ValueError):
        paired_cluster_bootstrap([0, 1], [.3, .6], [.4, .8], ["a", "a"],
                                 static_threshold=.5, temporal_threshold=.5)
    with pytest.raises(ValueError):
        paired_cluster_bootstrap([0, 1], [.3, .6], [.4, .8], ["a", "b"],
                                 static_threshold=.5, temporal_threshold=.5, repeats=5)


def test_complete_comparison_never_reads_source_test(tmp_path):
    rng = np.random.default_rng(11)
    val = pd.DataFrame(rng.normal(size=(48, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    val["sample_id"] = [f"v{i // 6}_f{i % 6}" for i in range(48)]
    val["video_id"] = [f"v{i // 6}" for i in range(48)]
    val["frame_index"] = [i % 6 for i in range(48)]
    val["timestamp_s"] = [(i % 6) * .5 for i in range(48)]
    val["source_fps"] = 30.0
    val["split"] = "val"
    val["error_knees_forward"] = [i % 2 for i in range(48)]
    val["error_knees_inward"] = [int(i % 5 == 0) for i in range(48)]
    val["any_of_two_annotated_errors"] = val[[
        "error_knees_forward", "error_knees_inward"]].max(axis=1)
    input_dir = tmp_path / "features"
    input_dir.mkdir()
    val.to_csv(input_dir / "fitness_aqa_val_pose.csv", index=False)
    (input_dir / "fitness_aqa_test_pose.csv").write_text("corrupted,test,file\n")

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    old = {
        "status": "train_group_calibrated_validation_evaluation_ONLY",
        "validation_frames": 48, "validation_videos": 8,
        "max_gap_seconds": .75, "seed": 42, "variants": {},
    }
    for variant in ("static", "causal_temporal"):
        x = features(val, temporal=variant != "static")
        old["variants"][variant] = {"tasks": {}}
        for target in ("error_knees_forward", "error_knees_inward"):
            clf = RandomForestClassifier(n_estimators=5, random_state=10)
            clf.fit(x, val[target])
            threshold = .3 if variant == "static" else .25
            joblib.dump({
                "model": clf, "target": target, "variant": variant,
                "schema_version": SCHEMA_VERSION if variant == "static" else TEMPORAL_SCHEMA,
                "feature_names": list(x.columns), "classes": [0, 1],
                "threshold_selected_from_train_only": threshold,
            }, model_dir / f"{variant}_{target}.joblib")
            old["variants"][variant]["tasks"][target] = {
                "train_calibrated_threshold": threshold
            }
    (model_dir / "comparison.json").write_text(json.dumps(old))
    output = tmp_path / "report" / "bootstrap.json"
    r = compare(input_dir, model_dir, output, repeats=25)
    assert r["validation_videos"] == 8
    assert r["tasks"]["error_knees_forward"]["frames"] == 48
    assert r["tasks"]["error_knees_inward"]["valid_bootstrap_samples"] >= 13
    assert output.is_file()
    assert not (tmp_path / "report" / "fitness_aqa_test_pose.csv").exists()


def test_refuses_stale_validation_dataset(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    val = pd.DataFrame(np.zeros((2, 103)), columns=FEATURE_NAMES)
    val["video_id"] = ["a", "b"]
    val["sample_id"] = ["a_0", "b_0"]
    val["split"] = "val"
    val["timestamp_s"] = [0.0, 0.0]
    val["frame_index"] = [0, 0]
    val["source_fps"] = 30.0
    val["error_knees_forward"] = [0, 1]
    val["error_knees_inward"] = [0, 1]
    val["any_of_two_annotated_errors"] = [0, 1]
    val.to_csv(feature_dir / "fitness_aqa_val_pose.csv", index=False)
    models = tmp_path / "models"
    models.mkdir()
    (models / "comparison.json").write_text(json.dumps({
        "status": "train_group_calibrated_validation_evaluation_ONLY",
        "validation_frames": 99, "validation_videos": 2,
    }))
    with pytest.raises(ValueError, match="Validation CSV count"):
        compare(feature_dir, models, tmp_path / "output.json")
