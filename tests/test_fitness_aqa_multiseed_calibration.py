"""Synthetic-only tests for train-group calibration robustness protocol."""
import json

import numpy as np
import pandas as pd
import pytest

from gym_pose.features import FEATURE_NAMES
from scripts.fitness_aqa_multiseed_calibration import compare, stable_summary


def fake_split(split, clips=16, per_clip=10):
    rng = np.random.default_rng(123 if split == "train" else 321)
    rows = []
    for clip in range(clips):
        for f in range(per_clip):
            r = dict(zip(FEATURE_NAMES, rng.normal(size=len(FEATURE_NAMES))))
            r.update({
                "sample_id": f"{split}_{clip}_{f}", "video_id": f"{split}_video_{clip}",
                "split": split, "timestamp_s": f / 2, "frame_index": f,
                "source_fps": 30.0,
                "error_knees_forward": int(clip % 3 == 0 and f % 2 == 0),
                "error_knees_inward": int(clip % 4 == 0 and f < 6),
            })
            r["any_of_two_annotated_errors"] = int(
                r["error_knees_forward"] or r["error_knees_inward"])
            rows.append(r)
    return pd.DataFrame(rows)


def test_stable_summary_ranges():
    rows = [
        {"calibration_seed": 11, "threshold_from_train_calibration": .2,
         "validation": {"macro_f1": .4, "precision_error": .2,
                        "recall_error": .6, "false_positive_rate": .2}},
        {"calibration_seed": 22, "threshold_from_train_calibration": .4,
         "validation": {"macro_f1": .5, "precision_error": .3,
                        "recall_error": .8, "false_positive_rate": .1}},
    ]
    s = stable_summary(rows)
    assert s["n_runs"] == 2
    assert s["threshold_range"] == pytest.approx([.2, .4])
    assert s["metrics"]["macro_f1"]["mean"] == pytest.approx(.45)


def test_train_only_multiseed_calibration_and_frozen_validation(tmp_path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    train = fake_split("train")
    val = fake_split("val", clips=8)
    train.to_csv(features_dir / "fitness_aqa_train_pose.csv", index=False)
    val.to_csv(features_dir / "fitness_aqa_val_pose.csv", index=False)
    # Any accidental test read should fail.
    (features_dir / "fitness_aqa_test_pose.csv").write_text("THIS,IS,NOT,A,VALID,CSV\n")
    output_dir = tmp_path / "out"
    report = compare(
        features_dir, output_dir,
        calibration_seeds=(42, 52, 62), forest_seed=42, trees=6,
    )
    assert report["status"] == "train_calibration_seed_stability_validation_descriptive_only"
    assert report["calibration_seeds"] == [42, 52, 62]
    assert set(report["variants"]) == {"static", "causal_temporal"}
    for variant in report["variants"].values():
        for target in variant["tasks"].values():
            assert len(target["runs"]) == 3
            assert target["seed_stability"]["n_runs"] == 3
            assert all(run["validation"]["n"] == len(val) for run in target["runs"])
            # Same full-train RF means AP cannot change with calibration seed.
            ap = target["validation_ap_invariant_across_calibration_seeds"]
            assert 0 <= ap <= 1
    for task in report["paired_seed_differences"].values():
        assert len(task["runs"]) == 3
        assert 0 <= task["temporal_macro_f1_greater_count"] <= 3
    saved = json.loads((output_dir / "multiseed_calibration.json").read_text())
    assert saved["forest_seed_fixed"] == 42
    assert not list(output_dir.glob("*.joblib"))


def test_duplicate_calibration_seeds_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        compare(tmp_path, tmp_path, calibration_seeds=(42, 42), trees=2)


def test_calibration_split_leakage_rejected(tmp_path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    train = fake_split("train")
    val = fake_split("val", clips=8)
    val.loc[0, "video_id"] = train.loc[0, "video_id"]
    train.to_csv(features_dir / "fitness_aqa_train_pose.csv", index=False)
    val.to_csv(features_dir / "fitness_aqa_val_pose.csv", index=False)
    with pytest.raises(ValueError, match="Leakage"):
        compare(features_dir, tmp_path / "out", calibration_seeds=(42, 52), trees=2)
