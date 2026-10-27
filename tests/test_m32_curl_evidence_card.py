"""M32: pure synthetic tests; never touch owner videos, features or TEST."""
from __future__ import annotations

import json

import pytest

from scripts import m32_curl_evidence_card as m32


def fake_m31():
    families = dict(m32.FAMILIES)
    reports = {}
    uncertainty = {}
    for cv, fold_count in m32.STRATEGIES.items():
        models = {}
        directions = {}
        intervals = {}
        for name in families:
            m = {
                "n": 90,
                "balanced_accuracy": .60,
                "f1_bad": .60,
                "recall_bad": .60,
                "average_precision_bad": .62,
            }
            models[name] = {
                "all_paired": m,
                "by_view": {
                    view: {**m, "n": 30}
                    for view in m32.VIEWS
                }
            }
            if name != "baseline_264":
                directions[name] = {
                    "better_subjects": 1,
                    "equal_subjects": 12,
                    "worse_subjects": 2,
                }
                intervals[name] = {
                    metric: {
                        "lower_95": -.07,
                        "median": -.02,
                        "upper_95": .03,
                        "fraction_positive": .20,
                    } for metric in ("balanced_accuracy", "recall_bad")
                }
        reports[cv] = {
            "folds": fold_count,
            "paired_clips": 90,
            "models": models,
            "subjects_with_better_or_worse_accuracy_vs_baseline": directions
        }
        uncertainty[cv] = {
            "resamples_requested": 1000,
            "deltas_vs_baseline_264": intervals
        }
    return {
        "m30_reproducibility": m32.PARITY,
        "train_subjects": 15,
        "paired_train_clips": 90,
        "source_video_decoded": False,
        "original_VAL_or_TEST_scored": False,
        "production_model_published": False,
        "feature_sets": families,
        "strategies": reports,
        "cluster_bootstrap": uncertainty
    }


def test_m32_publication_card_has_exact_baseline_and_no_private_values():
    content = m32.build_card(fake_m31())
    assert "# M32:" in content
    assert "Baseline 264" in content
    assert "kinematic_276" in content
    assert "tracking_proxy_268" in content
    assert "temporal_280" in content
    assert "Group5/LOSO" in content
    assert "15 frozen original M20 TRAIN" in content
    assert "0.5" in content
    assert "VAL/TEST" in content
    assert "-7.00 pp to +3.00 pp" in content
    assert "consent_ref" not in content
    assert "subject_id" not in content


def test_m32_refuses_holdout_and_wrong_cohorts():
    r = fake_m31()
    r["original_VAL_or_TEST_scored"] = True
    with pytest.raises(ValueError, match="TRAIN-only"):
        m32.build_card(r)

    r = fake_m31()
    r["train_subjects"] = 16
    with pytest.raises(ValueError, match="15-person"):
        m32.build_card(r)

    r = fake_m31()
    r["m30_reproducibility"] = "M30 report absent"
    with pytest.raises(ValueError, match="not verified"):
        m32.build_card(r)

    r = fake_m31()
    r["feature_sets"]["temporal_280"] = 281
    with pytest.raises(ValueError, match="feature-model"):
        m32.build_card(r)


def test_m32_refuses_invalid_uncertainty_and_camera_slices():
    r = fake_m31()
    r["cluster_bootstrap"]["loso"]["deltas_vs_baseline_264"][
        "kinematic_276"]["balanced_accuracy"]["upper_95"] = -.2
    with pytest.raises(ValueError, match="percentile order"):
        m32.build_card(r)

    r = fake_m31()
    r["strategies"]["loso"]["models"]["baseline_264"]["by_view"]["front"]["n"] = 40
    with pytest.raises(ValueError, match="camera slice"):
        m32.build_card(r)

    r = fake_m31()
    r["strategies"]["loso"]["subjects_with_better_or_worse_accuracy_vs_baseline"][
        "temporal_280"]["worse_subjects"] = 4
    with pytest.raises(ValueError, match="sum to 15"):
        m32.build_card(r)


def test_m32_local_cli_no_overwrite_and_aggregate_only(tmp_path):
    report = tmp_path / "m31_aggregate.json"
    report.write_text(json.dumps(fake_m31()), encoding="utf-8")
    target = tmp_path / "m32_card.md"
    arguments = [
        "--m31-report", str(report), "--output", str(target)
    ]
    assert m32.main(arguments) == 0
    assert target.is_file()
    assert "60.00%" in target.read_text(encoding="utf-8")
    assert m32.main(arguments) == 1
