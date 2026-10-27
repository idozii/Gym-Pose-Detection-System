"""M25 synthetic guards for participant-level LOSO and paired intervals."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from gym_pose import gym7_m25_loso as m25
from gym_pose.gym7_m24_cv import FEATURE_COLUMNS, COLS, META


def synthetic_tables(n_people=7):
    random = np.random.default_rng(17)
    common, expanded, manifest = [], [], []
    for p in range(n_people):
        subject = f"synthetic_person_{p}"
        for view in ("front", "side", "diagonal"):
            for label in ("good", "bad"):
                id_ = f"clip_{p}_{view}_{label}"
                m = {"video_id": id_, "subject_id": subject,
                     "exercise": "pushup", "label": label, "view": view}
                feat = random.normal(.3 if label == "bad" else -.3, 1.,
                                     len(FEATURE_COLUMNS))
                row = {**m, **dict(zip(FEATURE_COLUMNS, feat))}
                manifest.append({**m, "archive_member": f"{id_}.mp4"})
                expanded.append(row)
                if view != "front" or label == "good":
                    common.append(row)
    def frame(rows):
        return pd.DataFrame(rows, columns=COLS).set_index("video_id", drop=False)
    return frame(common), frame(expanded), manifest


def test_m25_cluster_bootstrap_is_paired_and_deterministic():
    truth = {"a": 0, "b": 1, "c": 0, "d": 1, "e": 0, "f": 1}
    persons = {"a": "p1", "b": "p1", "c": "p2", "d": "p2",
               "e": "p3", "f": "p3"}
    scores = {
        "full_paired": {"a": .2, "b": .8, "c": .2, "d": .8, "e": .2, "f": .8},
        "upper_paired": {"a": .9, "b": .2, "c": .9, "d": .2, "e": .9, "f": .2},
        "upper_expanded": {"a": .1, "b": .8, "c": .1, "d": .8, "e": .1, "f": .8},
    }
    # Require >=5 actual subject groups.
    persons.update({"e": "p5", "f": "p6"})
    persons.update({"c": "p3", "d": "p4"})
    r1 = m25._cluster_uncertainty(truth, scores, persons,
                                  n_resamples=100, seed=42)
    r2 = m25._cluster_uncertainty(truth, scores, persons,
                                  n_resamples=100, seed=42)
    assert r1 == r2
    assert r1["resamples_requested"] == 100
    reverse = r1["paired_delta_vs_full"]["upper_paired"]["delta_balanced_accuracy"]
    assert reverse["upper_95"] <= 0
    unchanged = r1["paired_delta_vs_full"]["upper_expanded"]["delta_balanced_accuracy"]
    assert unchanged["lower_95"] == 0
    assert unchanged["upper_95"] == 0
    with pytest.raises(ValueError, match="100..5000"):
        m25._cluster_uncertainty(truth, scores, persons, n_resamples=5)


def test_m25_loso_excludes_person_from_all_three_model_fits(monkeypatch):
    common, upper, original = synthetic_tables(n_people=7)
    people = {row["subject_id"] for row in original}
    monkeypatch.setattr(m25, "verify_cohort", lambda *_args: (
        sorted(common.index), sorted(upper.index), sorted(people), len(original)
    ))

    from types import SimpleNamespace
    train_subject_sets = []

    def fake_fit(data, *, n_trees):
        subjects = set(data.subject_id)
        train_subject_sets.append(subjects)
        return SimpleNamespace(subjects=subjects)

    def fake_predict(model, data):
        assert not set(data.subject_id) & model.subjects
        return np.array([.7 if label == "bad" else .3 for label in data.label])

    monkeypatch.setattr(m25, "_fit", fake_fit)
    monkeypatch.setattr(m25, "_predict", fake_predict)
    report = m25.evaluate_loso(
        common, upper, original, pd.DataFrame(),
        trees=25, bootstrap=100, seed=42
    )
    assert len(train_subject_sets) == len(people) * 3
    assert report["original_train_subjects_in_pushup"] == 7
    assert report["paired_common_clips"] == 35
    assert report["upper_total_admitted_clips"] == 42
    assert report["upper_only_recovered_clips"] == 7
    assert report["loso_folds"] == 7
    assert report["test_accessed"] is False
    assert report["source_video_decoded"] is False
    assert report["validation_or_test_subjects_scored"] is False
    for v in report["model_scores_on_identical_clips"].values():
        assert v["paired_common_loso"]["n"] == 35
        assert v["paired_common_loso"]["balanced_accuracy"] == 1.0
    dump = json.dumps(report)
    assert "synthetic_person" not in dump
    assert "clip_0_" not in dump


def test_m25_invalid_model_size_fails_without_fitting():
    full, upper, original = synthetic_tables(n_people=7)
    with pytest.raises(ValueError, match="original seed"):
        m25.evaluate_loso(full, upper, original, pd.DataFrame(), seed=12)
    with pytest.raises(ValueError, match="original seed"):
        m25.evaluate_loso(full, upper, original, pd.DataFrame(), trees=10)
