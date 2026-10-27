"""M31: private-safe synthetic tests for participant paired Curl robustness."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from scripts import m31_curl_paired_stability as m31
from scripts import m30_bicep_curl_temporal as m30


def fake_source(n_people=6):
    rng = np.random.default_rng(13)
    original, temporal, manifest = [], [], []
    for person in range(n_people):
        for view in ("front", "side", "diagonal"):
            for label in ("good", "bad"):
                vid = f"synthetic_curl_{person}_{view}_{label}"
                meta = {
                    "video_id": vid, "subject_id": f"person_{person:02d}",
                    "exercise": "bicep_curl", "label": label, "view": view
                }
                source = {**meta, "archive_member": f"fake/{vid}.mp4"}
                manifest.append(source)
                base = {
                    **source,
                    **dict(zip(m30.FEATURES, map(float, rng.normal(
                        loc=.35 if label == "bad" else -.35,
                        size=len(m30.FEATURES)
                    ))))
                }
                original.append(base)
                temporal.append({
                    **meta, **dict(zip(m30.MOTION_COLS, map(float, rng.normal(
                        loc=.2 if label == "bad" else -.2,
                        size=len(m30.MOTION_COLS)
                    ))))
                })
    return (
        pd.DataFrame(original),
        pd.DataFrame(temporal, columns=m30.MOTION_FILE_COLS),
        manifest
    )


def patch_sources(monkeypatch, original, manifest):
    monkeypatch.setattr(m31, "read_manifest", lambda _p: manifest)
    monkeypatch.setattr(m31, "load_clip_table", lambda _p: original)
    expected = sorted(original.video_id)
    monkeypatch.setattr(
        m31.m30, "frozen_curl_train",
        lambda _manifest, _original: (
            expected, original, {r["video_id"]: r for r in manifest}
        )
    )


def test_m31_train_only_exact_clips_and_participant_fold_separation(
        tmp_path, monkeypatch):
    original, motion, manifest = fake_source()
    patch_sources(monkeypatch, original, manifest)
    motion_path = tmp_path / "local_motion.csv"
    motion.to_csv(motion_path, index=False)
    models_fitted = []

    def predict_without_models(frame, features, training, heldout):
        fit_people = set(frame.iloc[training].subject_id)
        heldout_people = set(frame.iloc[heldout].subject_id)
        assert not fit_people & heldout_people
        assert len(fit_people | heldout_people) == 6
        assert len(features) in (264, 268, 276, 280)
        models_fitted.append(tuple(sorted(heldout_people)))
        return np.asarray([
            .8 if label == "bad" else .2
            for label in frame.iloc[heldout].label
        ])

    monkeypatch.setattr(m31, "_fit_predict", predict_without_models)
    report = m31.run_study(
        motion_path, tmp_path / "manifest.csv", tmp_path / "original.csv",
        resamples=100
    )
    assert report["train_subjects"] == 6
    assert report["paired_train_clips"] == 36
    assert report["strategies"]["group5"]["folds"] == 5
    assert report["strategies"]["loso"]["folds"] == 6
    assert len(models_fitted) == (5 + 6) * 4
    assert report["source_video_decoded"] is False
    assert report["original_VAL_or_TEST_scored"] is False
    assert report["production_model_published"] is False
    for strategy in ("group5", "loso"):
        assert all(
            report["strategies"][strategy]["models"][family][
                "all_paired"]["balanced_accuracy"
            ] == 1.0
            for family in m30.FAMILIES
        )
        for comparison in report["cluster_bootstrap"][strategy][
            "deltas_vs_baseline_264"
        ].values():
            for metric in comparison.values():
                assert metric["lower_95"] == 0.
                assert metric["upper_95"] == 0.
    text = json.dumps(report)
    assert "synthetic_curl_" not in text
    assert "person_" not in text
    assert "fake/" not in text


def test_m31_fails_closed_on_non_train_video_and_changed_identity(
        tmp_path, monkeypatch):
    original, motion, manifest = fake_source()
    patch_sources(monkeypatch, original, manifest)
    path = tmp_path / "motion.csv"

    missing = motion.iloc[:-1]
    missing.to_csv(path, index=False)
    with pytest.raises(ValueError, match="clip IDs differ"):
        m31._load_verified_train(path, tmp_path / "manifest",
                                 tmp_path / "original")

    leak = motion.copy()
    leak.loc[0, "subject_id"] = "unknown_person"
    leak.to_csv(path, index=False)
    with pytest.raises(ValueError, match="metadata mismatch"):
        m31._load_verified_train(path, tmp_path / "manifest",
                                 tmp_path / "original")

    wrong = motion.copy()
    wrong.loc[0, "label"] = "bad" if wrong.at[0, "label"] == "good" else "good"
    wrong.to_csv(path, index=False)
    with pytest.raises(ValueError, match="metadata mismatch"):
        m31._load_verified_train(path, tmp_path / "manifest",
                                 tmp_path / "original")


def test_m31_paired_bootstrap_is_deterministic_and_exposes_tradeoff():
    truth = {f"v{i}": i % 2 for i in range(12)}
    people = {f"v{i}": f"subject_{i // 2}" for i in range(12)}
    good = {k: .8 if val else .2 for k, val in truth.items()}
    bad = {k: .2 if val else .8 for k, val in truth.items()}
    probabilities = {
        "baseline_264": good,
        "kinematic_276": bad,
        "tracking_proxy_268": good,
        "temporal_280": good
    }
    a = m31.paired_bootstrap(truth, probabilities, people,
                             resamples=100, seed=42)
    b = m31.paired_bootstrap(truth, probabilities, people,
                             resamples=100, seed=42)
    assert a == b
    assert a["resamples_requested"] == 100
    reverse = a["deltas_vs_baseline_264"]["kinematic_276"][
        "balanced_accuracy"
    ]
    assert reverse["upper_95"] <= 0
    same = a["deltas_vs_baseline_264"]["temporal_280"][
        "balanced_accuracy"
    ]
    assert same["lower_95"] == same["upper_95"] == 0.0
    with pytest.raises(ValueError, match="100..5000"):
        m31.paired_bootstrap(
            truth, probabilities, people, resamples=10
        )
    malformed = dict(probabilities)
    malformed["tracking_proxy_268"] = {"v0": .2}
    with pytest.raises(ValueError, match="same paired TRAIN videos"):
        m31.paired_bootstrap(
            truth, malformed, people, resamples=100
        )


def test_m31_m30_reproducibility_guard(tmp_path):
    metrics = {
        "balanced_accuracy": .60,
        "f1_bad": .6087,
        "recall_bad": .6222,
        "average_precision_bad": .6132
    }
    report = {
        "paired_train_clips": 90, "train_subjects": 15,
        "strategies": {
            strategy: {
                "folds": folds,
                "models": {
                    name: {"all_paired": dict(metrics)}
                    for name in m30.FAMILIES
                }
            } for strategy, folds in (("group5", 5), ("loso", 15))
        }
    }
    old = {
        "paired_clips": 90, "subjects": 15,
        "models": {
            strategy: {
                "folds": folds,
                "models": {
                    name: {"all_paired": dict(metrics)}
                    for name in m30.FAMILIES
                }
            } for strategy, folds in (("group5", 5), ("loso", 15))
        }
    }
    path = tmp_path / "m30_report.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    assert "reproduced exactly" in m31.verify_m30_parity(report, path)
    old["models"]["loso"]["models"]["temporal_280"][
        "all_paired"
    ]["balanced_accuracy"] = .65
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(ValueError, match="failed to reproduce M30"):
        m31.verify_m30_parity(report, path)


def test_m31_real_small_rf_with_publisher_labels(tmp_path, monkeypatch):
    """Run sklearn codepath once; synthetic rows only, never user video."""
    original, motion, manifest = fake_source()
    patch_sources(monkeypatch, original, manifest)
    path = tmp_path / "motion.csv"
    motion.to_csv(path, index=False)
    monkeypatch.setattr(m31, "TREES", 15)
    report = m31.run_study(path, tmp_path / "manifest",
                           tmp_path / "original", resamples=100)
    for strategy in ("group5", "loso"):
        for family in m30.FAMILIES:
            assert report["strategies"][strategy]["models"][family][
                "all_paired"]["n"
            ] == 36
