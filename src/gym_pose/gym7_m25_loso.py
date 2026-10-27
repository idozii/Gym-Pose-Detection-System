"""M25 - Leave-One-Subject-Out and subject-cluster uncertainty for Push-up.

Offline exploratory evidence only. NO original VAL/TEST subject scoring,
source video decoding or joblib model publishing. Per-person predictions
are kept in memory only; output JSON holds aggregate metrics and
approximate, paired subject-cluster bootstrap uncertainty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from gym_pose.gym7_research import read_manifest
from gym_pose.gym7_training import load_clip_table
from gym_pose.gym7_m24_cv import (
    _fit, _metrics, _predict, _read_table, verify_cohort, MODELS,
)

SEED = 42


def _balanced_accuracy(y, p):
    """No sklearn warnings for bootstrap single-class resamples."""
    truth = np.asarray(y, dtype=int)
    pred = np.asarray(p, dtype=float) >= .5
    if set(truth) != {0, 1}:
        return np.nan
    return float(.5 * (
        np.mean(pred[truth == 1]) + np.mean(~pred[truth == 0])
    ))


def _bad_recall(y, p):
    truth = np.asarray(y, dtype=int)
    if not np.any(truth == 1):
        return np.nan
    return float(np.mean((np.asarray(p) >= .5)[truth == 1]))


def _cluster_uncertainty(truth_by_id, scores, subject_by_id, *,
                         n_resamples=1000, seed=SEED):
    """Subject bootstrap on fixed OOF predictions (NOT model retraining).

    Same sampled people and matched clips are used across every model.
    The 95% percentile interval is descriptive due to 14 people only.
    """
    if not 100 <= n_resamples <= 5000:
        raise ValueError("Cluster bootstrap resamples must be 100..5000")
    videos = sorted(truth_by_id)
    if any(set(scores[m]) != set(videos) for m in MODELS):
        raise ValueError("All models must predict the identical common videos")
    people = sorted({subject_by_id[ident] for ident in videos})
    if len(people) < 5:
        raise ValueError("At least five source participants needed")
    group = {
        person: [vid for vid in videos if subject_by_id[vid] == person]
        for person in people
    }
    rng = np.random.default_rng(seed)
    comparisons = {
        model: {"delta_balanced_accuracy": [], "delta_recall_bad": []}
        for model in ("upper_paired", "upper_expanded")
    }
    skipped = 0
    for _ in range(n_resamples):
        chosen = rng.choice(people, size=len(people), replace=True)
        selected = [ident for person in chosen for ident in group[person]]
        y = [truth_by_id[i] for i in selected]
        if set(y) != {0, 1}:
            skipped += 1
            continue
        full = [scores["full_paired"][i] for i in selected]
        reference_bal = _balanced_accuracy(y, full)
        reference_recall = _bad_recall(y, full)
        for model in comparisons:
            alt = [scores[model][i] for i in selected]
            comparisons[model]["delta_balanced_accuracy"].append(
                _balanced_accuracy(y, alt) - reference_bal
            )
            comparisons[model]["delta_recall_bad"].append(
                _bad_recall(y, alt) - reference_recall
            )
    result = {}
    for model, values in comparisons.items():
        result[model] = {}
        for metric, sampled in values.items():
            if not sampled:
                result[model][metric] = None
                continue
            lo, hi = np.quantile(sampled, [.025, .975])
            result[model][metric] = {
                "lower_95": float(lo),
                "upper_95": float(hi),
                "bootstrap_median": float(np.median(sampled)),
                "fraction_positive": float(np.mean(np.asarray(sampled) > 0)),
            }
    return {
        "resamples_requested": n_resamples,
        "usable_resamples": n_resamples - skipped,
        "skipped_one_label_resamples": skipped,
        "interval_type": (
            "Approximate 95% percentile interval from resampling "
            "original TRAIN subjects on fixed LOSO OOF predictions. "
            "Does not include RF retraining uncertainty; NOT "
            "an independent validation confidence interval."
        ),
        "paired_delta_vs_full": result,
    }


def evaluate_loso(full, upper, manifest_rows, original_features, *,
                  trees=200, bootstrap=1000, seed=SEED):
    if seed != SEED or not 25 <= trees <= 1000:
        raise ValueError("M25 requires original seed=42, trees between 25 and 1000")
    common_ids, upper_ids, subjects, original_clips = verify_cohort(
        full, upper, manifest_rows, original_features
    )
    predicted = {model: {} for model in MODELS}
    recovered = {}
    y_true = {vid: int(full.at[vid, "label"] == "bad") for vid in common_ids}
    person = {vid: full.at[vid, "subject_id"] for vid in common_ids}

    # LOSO uses all available original TRAIN participants and keeps all
    # viewpoints from each heldout person out of every model's training data.
    for heldout in subjects:
        training_common_ids = [
            vid for vid in common_ids if full.at[vid, "subject_id"] != heldout
        ]
        heldout_common_ids = [
            vid for vid in common_ids if full.at[vid, "subject_id"] == heldout
        ]
        training_expanded_ids = [
            vid for vid in upper_ids if upper.at[vid, "subject_id"] != heldout
        ]
        heldout_recovered_ids = sorted(
            vid for vid in upper_ids
            if upper.at[vid, "subject_id"] == heldout
            and vid not in full.index
        )
        if not heldout_common_ids:
            raise ValueError("Every cohort participant must have full-gate clips")
        for candidate in training_expanded_ids:
            if upper.at[candidate, "subject_id"] == heldout:
                raise AssertionError("Upper-expanded LOSO data leakage")
        fit_data = {
            "full_paired": full.loc[training_common_ids],
            "upper_paired": upper.loc[training_common_ids],
            "upper_expanded": upper.loc[training_expanded_ids],
        }
        eval_data = {
            "full_paired": full.loc[heldout_common_ids],
            "upper_paired": upper.loc[heldout_common_ids],
            "upper_expanded": upper.loc[heldout_common_ids],
        }
        for model_name in MODELS:
            fitted = _fit(fit_data[model_name], n_trees=trees)
            probs = _predict(fitted, eval_data[model_name])
            for vid, score in zip(heldout_common_ids, probs):
                if vid in predicted[model_name]:
                    raise AssertionError("Repeated out-of-fold prediction")
                predicted[model_name][vid] = float(score)
            if model_name == "upper_expanded" and heldout_recovered_ids:
                extra_probs = _predict(
                    fitted, upper.loc[heldout_recovered_ids]
                )
                for vid, score in zip(heldout_recovered_ids, extra_probs):
                    if vid in recovered:
                        raise AssertionError("Repeated recovered prediction")
                    recovered[vid] = (
                        int(upper.at[vid, "label"] == "bad"),
                        float(score),
                        str(upper.at[vid, "view"]),
                    )

    if any(set(d) != set(common_ids) for d in predicted.values()):
        raise AssertionError("Paired LOSO predictions missing")
    if set(recovered) != set(upper_ids) - set(common_ids):
        raise AssertionError("Recovered clip scoring missing")

    metrics = {}
    for model_name, probs in predicted.items():
        paired = [(y_true[vid], probs[vid]) for vid in common_ids]
        metrics[model_name] = {
            "paired_common_loso": _metrics(paired),
            "by_view": {
                view: _metrics([
                    (y_true[vid], probs[vid]) for vid in common_ids
                    if full.at[vid, "view"] == view
                ])
                for view in ("front", "side", "diagonal")
            },
        }
    recovered_pairs = [(truth, prob) for truth, prob, _ in recovered.values()]
    uncertainty = _cluster_uncertainty(
        y_true, predicted, person, n_resamples=bootstrap, seed=seed
    )
    return {
        "study": "M25 original TRAIN Leave-One-Subject-Out Push-up robustness",
        "dataset": "Mendeley Gym7 v3",
        "label_semantics": "video-level publisher good/bad, not named form errors",
        "original_m20_split_seed": seed,
        "original_train_subjects_in_pushup": len(subjects),
        "original_train_pushup_clips": original_clips,
        "paired_common_clips": len(common_ids),
        "upper_total_admitted_clips": len(upper_ids),
        "upper_only_recovered_clips": len(recovered),
        "loso_folds": len(subjects),
        "random_forest_trees": trees,
        "validation_or_test_subjects_scored": False,
        "source_video_decoded": False,
        "test_accessed": False,
        "model_scores_on_identical_clips": metrics,
        "upper_only_unpaired_loso": {
            "n": len(recovered),
            "all_views": _metrics(recovered_pairs),
            "by_view": {
                view: _metrics([
                    (truth, prob)
                    for truth, prob, camera in recovered.values()
                    if camera == view
                ])
                for view in ("front", "side", "diagonal")
            },
            "comparison_caution": (
                "Full-body gate cannot score these clips, so they "
                "cannot support a paired accuracy improvement claim."
            ),
        },
        "cluster_bootstrap": uncertainty,
        "caution": (
            "TRAIN-only LOSO and clustered bootstrap on fixed predictions "
            "are exploratory uncertainty checks. Higher pose coverage "
            "is not higher form-label accuracy. Freeze candidate before "
            "independent M20 VAL evaluation; never tune on TEST."
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", type=Path, default=Path(
        "data/features/m23_pushup_train_upper_v1/train_full_body.csv"))
    p.add_argument("--upper", type=Path, default=Path(
        "data/features/m23_pushup_train_upper_v1/train_upper_body.csv"))
    p.add_argument("--original-features", type=Path, default=Path(
        "data/features/m20_gym7/m20_clip_features.csv"))
    p.add_argument("--manifest", type=Path, default=Path(
        "data/external/mendeley_gym7/m20_manifest.csv"))
    p.add_argument("--output", type=Path, default=Path(
        "data/processed/m25_pushup_loso_robustness.json"))
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=1000)
    a = p.parse_args(argv)
    try:
        if a.output.exists():
            raise FileExistsError("M25 report exists; it will not be overwritten")
        original = load_clip_table(a.original_features)
        manifest = read_manifest(a.manifest)
        report = evaluate_loso(
            _read_table(a.full), _read_table(a.upper),
            manifest, original, trees=a.trees, bootstrap=a.bootstrap
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, AssertionError) as exc:
        print(f"M25 analysis failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
