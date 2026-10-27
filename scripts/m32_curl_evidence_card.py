"""Generate an honest, aggregate-only M32 Bicep Curl research card from M31.

Reads NO videos, landmarks, features, subjects IDs, M20 VAL/TEST or remote
data; never fits a model. Uses only the already verified M31 local JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

FAMILIES = {
    "baseline_264": 264,
    "kinematic_276": 276,
    "tracking_proxy_268": 268,
    "temporal_280": 280,
}
STRATEGIES = {"group5": 5, "loso": 15}
VIEWS = ("front", "side", "diagonal")
PARITY = "Original M30 GROUP5 and LOSO model metrics reproduced exactly"
METRIC_KEYS = ("balanced_accuracy", "f1_bad", "recall_bad",
               "average_precision_bad")


def _percent(x):
    if not isinstance(x, (int, float)) or not 0 <= x <= 1:
        raise ValueError("Invalid metric outside [0, 1]")
    return f"{x * 100:.2f}%"


def _pp(x):
    if not isinstance(x, (int, float)) or not -1 <= x <= 1:
        raise ValueError("Invalid paired delta")
    return f"{x * 100:+.2f} pp"


def validate(report):
    if report.get("m30_reproducibility") != PARITY:
        raise ValueError("M30 reproducibility was not verified")
    if (report.get("train_subjects") != 15
        or report.get("paired_train_clips") != 90
        or report.get("original_VAL_or_TEST_scored") is not False
        or report.get("source_video_decoded") is not False
        or report.get("production_model_published") is not False):
        raise ValueError("M32 only supports frozen 15-person/90-clip TRAIN-only study")
    if report.get("feature_sets") != FAMILIES:
        raise ValueError("M32 unexpected feature-model families")
    if set(report.get("strategies", {})) != set(STRATEGIES):
        raise ValueError("Incomplete original M30 CV strategy")
    if set(report.get("cluster_bootstrap", {})) != set(STRATEGIES):
        raise ValueError("Missing original paired cluster bootstrap")
    for cv, folds in STRATEGIES.items():
        item = report["strategies"][cv]
        if item.get("folds") != folds or item.get("paired_clips") != 90:
            raise ValueError("Frozen CV population or split count differs")
        if set(item.get("models", {})) != set(FAMILIES):
            raise ValueError("M32 requires all four original models")
        if set(item.get("subjects_with_better_or_worse_accuracy_vs_baseline", {})) != (
            set(FAMILIES) - {"baseline_264"}
        ):
            raise ValueError("Missing anonymized subject comparisons")
        bootstrap = report["cluster_bootstrap"][cv]
        if bootstrap.get("resamples_requested") != 1000:
            raise ValueError("M32 requires frozen 1000 bootstrap draws")
        deltas = bootstrap.get("deltas_vs_baseline_264", {})
        if set(deltas) != set(FAMILIES) - {"baseline_264"}:
            raise ValueError("Missing paired bootstrap delta")
        for name, model in item["models"].items():
            m = model["all_paired"]
            if m["n"] != 90:
                raise ValueError("Unequal paired video denominator")
            for metric in METRIC_KEYS:
                _percent(m[metric])
            views = model["by_view"]
            if set(views) != set(VIEWS) or sum(views[v]["n"] for v in VIEWS) != 90:
                raise ValueError("Incomplete or overlapping camera slice")
            for view in VIEWS:
                m_view = views[view]
                for metric in ("f1_bad", "recall_bad"):
                    _percent(m_view[metric])
                if m_view["balanced_accuracy"] is not None:
                    _percent(m_view["balanced_accuracy"])
        for name in set(FAMILIES) - {"baseline_264"}:
            counts = item["subjects_with_better_or_worse_accuracy_vs_baseline"][name]
            if set(counts) != {"better_subjects", "equal_subjects", "worse_subjects"}:
                raise ValueError("Incomplete subject comparison keys")
            if any(type(v) is not int or v < 0 for v in counts.values()):
                raise ValueError("Invalid subject comparison count")
            if sum(counts.values()) != 15:
                raise ValueError("Expected 15 anonymized participant outcomes")
            for metric in ("balanced_accuracy", "recall_bad"):
                values = deltas[name][metric]
                if set(values) != {
                    "lower_95", "median", "upper_95", "fraction_positive"
                }:
                    raise ValueError("Incomplete bootstrap uncertainty fields")
                lo, med, hi = (values[k] for k in (
                    "lower_95", "median", "upper_95"
                ))
                if not all(isinstance(v, (int, float)) and -1 <= v <= 1
                           for v in (lo, med, hi)) or not lo <= med <= hi:
                    raise ValueError("Invalid bootstrap percentile order")
                _percent(values["fraction_positive"])


def build_card(report):
    validate(report)
    strategies = report["strategies"]
    intervals = report["cluster_bootstrap"]
    lines = [
        "# M32: Bicep Curl baseline research evidence card",
        "",
        "**Research-only; no form correctness, medical or safety claim.**",
        "",
        "Source: Mendeley Gym7 v3 publisher whole-video good/bad labels.",
        "Cohort: 90 matched source videos from 15 frozen original M20 TRAIN people.",
        "Verification: M30 Group5/LOSO fully reproduced by M31.",
        "The original M20 VAL and TEST remain unscored by M31/M32.",
        "",
        "## Original participant-disjoint results",
        "",
        "| Variant | Features | Group5 BA | LOSO BA | LOSO bad recall | LOSO bad AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"
    ]
    for name, width in FAMILIES.items():
        group = strategies["group5"]["models"][name]["all_paired"]
        loso = strategies["loso"]["models"][name]["all_paired"]
        lines.append(
            f"| {name} | {width} | {_percent(group['balanced_accuracy'])} | "
            f"{_percent(loso['balanced_accuracy'])} | "
            f"{_percent(loso['recall_bad'])} | "
            f"{_percent(loso['average_precision_bad'])} |"
        )
    lines.extend([
        "",
        "## Paired participant bootstrap differences versus Baseline 264",
        "",
        "Intervals are descriptive 95% percentile estimates over 1,000 "
        "resamples of participants using *fixed* OOF predictions. "
        "They omit model refitting uncertainty and do not establish "
        "significance on independent users.",
        "",
        "| CV | Model | Delta BA 95% interval | Delta bad recall 95% interval |",
        "| --- | --- | --- | --- |"
    ])
    for cv in STRATEGIES:
        for name in list(FAMILIES)[1:]:
            diff = intervals[cv]["deltas_vs_baseline_264"][name]
            ba, recall = diff["balanced_accuracy"], diff["recall_bad"]
            lines.append(
                f"| {cv} | {name} | "
                f"{_pp(ba['lower_95'])} to {_pp(ba['upper_95'])} | "
                f"{_pp(recall['lower_95'])} to {_pp(recall['upper_95'])} |"
            )
    lines.extend([
        "",
        "## Participant-level LOSO outcomes",
        "",
        "| Model | More videos correct for subject | Equal | Fewer videos correct |",
        "| --- | ---: | ---: | ---: |"
    ])
    for name in list(FAMILIES)[1:]:
        counts = strategies["loso"][
            "subjects_with_better_or_worse_accuracy_vs_baseline"
        ][name]
        lines.append(
            f"| {name} | {counts['better_subjects']} | "
            f"{counts['equal_subjects']} | {counts['worse_subjects']} |"
        )
    lines.extend([
        "",
        "## Original Baseline 264: LOSO camera-view slices",
        "",
        "| Camera view | Number of clips | BA | F1 bad | Recall bad |",
        "| --- | ---: | ---: | ---: | ---: |"
    ])
    for view, m in (
        (view, strategies["loso"]["models"]["baseline_264"]["by_view"][view])
        for view in VIEWS
    ):
        ba = "n/a" if m["balanced_accuracy"] is None else _percent(m["balanced_accuracy"])
        lines.append(
            f"| {view} | {m['n']} | {ba} | "
            f"{_percent(m['f1_bad'])} | {_percent(m['recall_bad'])} |"
        )
    lines.extend([
        "",
        "## Frozen research reference",
        "",
        "Keep the original M20 264-feature *video-level* representation "
        "as Baseline 264, with RandomForestClassifier, 300 trees, "
        "min_samples_leaf=2, class_weight=balanced, random_state=42, "
        "n_jobs=1 and threshold 0.5. This is an experimental "
        "reference, not a model deployed to the website.",
        "",
        "The observed M31 LOSO Balanced Accuracy is 60.00%. "
        "Augmented variants did not consistently outperform it "
        "on source participants. Each paired LOSO BA bootstrap "
        "interval includes zero in the supplied results.",
        "",
        "Do not tune on previously inspected M20 VAL or open original "
        "TEST to chase improvement. Future assertions require genuinely "
        "independent consented participants, stable publisher-compatible "
        "annotations, camera parity checks and reported abstention.",
        "",
        "M32 reads only aggregate local M31 JSON, not video bytes, "
        "video-level predictions, personal IDs or original VAL/TEST.",
        ""
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m31-report", type=Path, default=Path(
        "data/processed/m31_curl_paired_subject_stability.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/m32_bicep_curl_research_card.md"))
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("Original M32 card exists and cannot be overwritten")
        report = json.loads(args.m31_report.read_text(encoding="utf-8"))
        content = build_card(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"M32 evidence card stopped: {error}", file=sys.stderr)
        return 1
    print(f"Written M32 research evidence card: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
