"""Repeatable RF stability checks using synthetic, no original source TEST data."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gym_pose.ohp_task_features import FEATURES_BY_TASK, TASKS
from scripts import assess_fitness_aqa_ohp_seed_stability as stability


def _synthetic_frames():
    rng = np.random.default_rng(601)
    result = {}
    for task_i, target in enumerate(TASKS):
        result[target] = {}
        for split, n_clips in (("train", 48), ("val", 16)):
            data = []
            for i in range(n_clips):
                video_id = f"{task_i+1}{1000 if split=='train' else 2000+i}" if False else (
                    f"{task_i + 1}_{split}_{i:03d}")
                is_positive = (i % 3 == 0) if target == "error_elbows" else (i % 2 == 0)
                for idx in range(3):
                    row = {
                        "video_id": video_id, "split": split,
                        "sample_id": f"{video_id}_f{idx:07d}",
                        "timestamp_s": idx * .5, "frame_index": idx,
                        target: int(is_positive),
                    }
                    for f in FEATURES_BY_TASK[target]:
                        row[f] = float(rng.normal(loc=.3 if is_positive else .15, scale=.17))
                    data.append(row)
            result[target][split] = pd.DataFrame(data)
    summary = {
        "splits": {
            split: {"tasks": {
                target: {"pose_coverage_eligible": (.315 if target=="error_elbows" else .392)}
                for target in TASKS}}
            for split in ("train", "val")
        },
    }
    return result, summary


def test_seed_parser_and_summary():
    assert stability._parse_seeds("41, 42, 43") == (41, 42, 43)
    for invalid in ("", "1", "2,2", "1,word", "-1,2", "1,4294967295"):
        with pytest.raises(ValueError):
            stability._parse_seeds(invalid)
    dist = stability._distribution([.1, .3, .5])
    assert dist["median"] == .3
    assert dist["min"] == .1 and dist["max"] == .5
    assert dist["sample_sd"] > 0
    with pytest.raises(ValueError):
        stability._distribution([float("nan")])


def test_multiseed_is_reproducible_and_uses_exact_val_frame_populations(
        tmp_path, monkeypatch):
    frames, source = _synthetic_frames()
    called = []
    def read_only_train_val(_directory, *, require_full=True):
        called.append(require_full)
        return frames, source
    monkeypatch.setattr(stability, "_load_data", read_only_train_val)
    a = stability.run(tmp_path / "features", tmp_path / "seed_a.json",
                      seeds=(41, 42, 43), trees=12, require_full=False)
    b = stability.run(tmp_path / "features", tmp_path / "seed_b.json",
                      seeds=(41, 42, 43), trees=12, require_full=False)
    assert a == b
    assert called == [False, False]
    assert not a["publisher_test_accessed"]
    assert a["source_train_group_calibration_only"]
    assert (tmp_path / "seed_a.json").exists()
    assert set(a["tasks"]) == set(TASKS)
    for target in TASKS:
        task = a["tasks"][target]
        assert len(task["results_by_seed"]) == 3
        assert task["validation_positive_videos"] >= 5
        assert task["validation_frames"] == len(frames[target]["val"])
        assert task["stability_summary"]["temporal_minus_static"][
            "average_precision"]["seeds_positive"] <= 3
        for one in task["results_by_seed"]:
            assert one["video_groups_disjoint"]
            assert one["fit_positive_video_count"] >= 10
            assert one["cal_positive_video_count"] >= 3
            for variant in ("static", "causal_temporal"):
                entry = one["variants"][variant]
                assert entry["feature_count"] == (30 if variant == "static" else 42)
                assert 0 < entry["threshold_from_train_groups_only"] < 1
                assert 0 <= entry["validation_average_precision"] <= 1
                assert 0 <= entry["validation_train_calibrated"]["macro_f1"] <= 1
    assert not list(tmp_path.glob("*.joblib")), "Robustness script must not alter saved baseline models"


def test_bad_config_is_rejected_before_source_load(tmp_path, monkeypatch):
    monkeypatch.setattr(stability, "_load_data", lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(AssertionError("source should never load"))))
    for opts in (
        {"seeds": (41,)}, {"seeds": (42,42)}, {"trees": 0},
        {"max_gap_seconds": float("nan")}, {"seeds": (-1,2)},
    ):
        with pytest.raises(ValueError):
            stability.run(tmp_path, tmp_path / "report.json", **opts)


def test_summary_is_descriptive_not_unbiased_confidence(tmp_path, monkeypatch):
    monkeypatch.setattr(stability, "_load_data", lambda *_a, **_k: _synthetic_frames())
    result = stability.run(tmp_path, tmp_path / "seeds.json",
                           seeds=(41, 42), trees=7, require_full=False)
    assert any("exploratory" in item.lower() for item in result["limitations"])
    assert any("NOT independent" in item for item in result["limitations"])
    for task in result["tasks"].values():
        for label, dist in task["stability_summary"]["temporal_minus_static"].items():
            assert (dist["seeds_positive"] + dist["seeds_equal"] +
                    dist["seeds_negative"]) == 2
            assert dist["distribution"]["min"] <= dist["distribution"]["max"]
