"""M26 - predeclared M20 validation study for versioned Push-up clip quality.

This is offline research, not a website endpoint. The original M20 split
is reconstructed BEFORE selecting any M26 videos or fitting any model.
Only original M20 VAL-participant Push-up videos may be decoded and scored.
TEST participant videos are NEVER opened, decoded, fitted, or evaluated.

The frozen comparison is 200-tree min_leaf=2 class-balanced RandomForest,
threshold=.5, full-paired training versus upper-expanded training,
both scored on identical VAL clip IDs admitted by both policies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

from gym_pose.gym7_research import read_manifest
from gym_pose.gym7_training import load_clip_table, split_by_subject
from gym_pose.gym7_m24_cv import _fit, _metrics, _predict, _read_table
from scripts.m23_train_only_pushup_features import (
    _extract_video, FEATURE_COLUMNS, META, COLS, GATES, SCHEMA,
)

SEED = 42
TREES = 200
TRAIN_SETS = ("full_paired", "upper_expanded")


def frozen_splits(manifest, original_features):
    parts = split_by_subject(original_features, seed=SEED)
    people = {split: set(df.subject_id) for split, df in parts.items()}
    if (people["train"] & people["val"] or people["train"] & people["test"]
        or people["val"] & people["test"]):
        raise AssertionError("M20 person-level holdout separation failed")
    by_video = {record["video_id"]: record for record in manifest}
    if len(by_video) != len(manifest):
        raise ValueError("Duplicate publisher manifest video_id")
    for record in original_features.itertuples(index=False):
        source = by_video.get(record.video_id)
        if (source is None or
            any(getattr(record, key) != source[key]
                for key in ("video_id", "subject_id", "exercise", "label", "view"))):
            raise ValueError("M20 original accepted feature metadata disagrees with manifest")
    rows = {
        split: sorted(
            (r for r in manifest
             if r["subject_id"] in people[split] and r["exercise"] == "pushup"),
            key=lambda row: row["video_id"]
        )
        for split in ("train", "val", "test")
    }
    if not rows["train"] or not rows["val"] or not rows["test"]:
        raise ValueError("Missing Push-up subject split; do not recreate splits")
    return people, rows


def _verify_gate_tables(full, upper, manifest_rows, subjects, *, split):
    source = {row["video_id"]: row for row in manifest_rows}
    if len(source) != len(manifest_rows):
        raise ValueError("Duplicate original Push-up video ID")
    if not set(full.index).issubset(set(upper.index)):
        raise ValueError("Full-gate admitted clips must be in upper-gate CSV")
    if not len(full) or not len(upper):
        raise ValueError("Both gates require nonempty accepted feature tables")
    if not set(upper.index).issubset(source):
        raise ValueError(f"{split} contains unknown or non-{split} video ID")
    for gate, table in (("full", full), ("upper", upper)):
        for row in table.itertuples(index=False):
            identity = source[row.video_id]
            if (any(getattr(row, key) != identity[key] for key in META)
                or row.subject_id not in subjects):
                raise ValueError(f"{split} {gate} row metadata or person mismatch")
    return source


def extract_validation(archive: Path, manifest_path: Path, original_csv: Path,
                       destination: Path, *, max_video_mb=500,
                       sample_fps=2., max_frames=240):
    if destination.exists():
        raise FileExistsError("M26 validation feature destination exists; do not overwrite")
    if max_video_mb < 5 or not 0 < sample_fps <= 10 or not 4 <= max_frames <= 2000:
        raise ValueError("Invalid original M23 video extraction settings")
    manifest = read_manifest(manifest_path)
    original = load_clip_table(original_csv)
    people, split_rows = frozen_splits(manifest, original)
    val = split_rows["val"]

    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("Activate the dedicated .venv-m20 camera environment") from exc

    admitted = {gate: [] for gate in GATES}
    by_label_view = {gate: Counter() for gate in GATES}
    failures = Counter()
    with zipfile.ZipFile(archive) as zipped:
        archive_members = {x.filename: x for x in zipped.infolist()}
        with mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=1,
            min_detection_confidence=.5
        ) as detector:
            for record in val:
                # Only original M20 VAL identities reach this ZIP accessor.
                outcome = _extract_video(
                    zipped, archive_members.get(record["archive_member"]),
                    detector, cv2, sample_fps=sample_fps,
                    max_frames=max_frames, max_video_mb=max_video_mb
                )
                if "error" in outcome:
                    failures[outcome["error"]] += 1
                    continue
                for gate in GATES:
                    vector = outcome["clip_vectors"][gate]
                    if vector is None:
                        continue
                    admitted[gate].append({
                        **{name: record[name] for name in META},
                        **dict(zip(FEATURE_COLUMNS, map(float, vector)))
                    })
                    by_label_view[gate][(record["label"], record["view"])] += 1

    if not admitted["full_body"] or not admitted["upper_body"]:
        raise ValueError("Original VAL has no accepted clips for one of the gates")
    destination.mkdir(parents=True, exist_ok=False)
    for gate, rows in admitted.items():
        with (destination / f"val_{gate}.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLS)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "study": "M26 M20-frozen VAL-only Push-up video extraction",
        "schema": SCHEMA,
        "original_m20_seed": SEED,
        "validation_participants": len({r["subject_id"] for r in val}),
        "validation_manifest_clips": len(val),
        "full_gate_accepted": len(admitted["full_body"]),
        "upper_gate_accepted": len(admitted["upper_body"]),
        "by_label_and_view": {
            gate: {f"{label}/{view}": value
                   for (label, view), value in sorted(by_label_view[gate].items())}
            for gate in GATES
        },
        "file_failures": dict(failures),
        "original_TEST_video_decoded": False,
        "original_TRAIN_video_decoded": False,
        "notice": (
            "Validation has been consulted earlier during M20 research. "
            "This is a held-out-from-NEW-schema comparison, not a previously "
            "untouched independent external validation. Keep final TEST sealed."
        ),
    }
    (destination / "val_extract_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def evaluate_validation(original_csv: Path, manifest_path: Path,
                        train_full_path: Path, train_upper_path: Path,
                        val_full_path: Path, val_upper_path: Path):
    manifest = read_manifest(manifest_path)
    original = load_clip_table(original_csv)
    people, rows = frozen_splits(manifest, original)
    train_full = _read_table(train_full_path)
    train_upper = _read_table(train_upper_path)
    val_full = _read_table(val_full_path)
    val_upper = _read_table(val_upper_path)

    _verify_gate_tables(train_full, train_upper,
                        rows["train"], people["train"], split="TRAIN")
    _verify_gate_tables(val_full, val_upper,
                        rows["val"], people["val"], split="VAL")

    paired = sorted(val_full.index)
    upper_only = sorted(set(val_upper.index) - set(paired))
    # Guard against changing the declared comparison population after
    # receiving different model scores.
    if not paired:
        raise ValueError("Cannot evaluate no common validation clips")
    if (set(train_full.subject_id) | set(train_upper.subject_id)) & (
        set(val_full.subject_id) | set(val_upper.subject_id)
    ):
        raise AssertionError("Training and validation participant leakage")
    training = {
        "full_paired": train_full,
        "upper_expanded": train_upper,
    }
    val_tables = {
        "full_paired": val_full,
        "upper_expanded": val_upper.loc[paired],
    }

    report = {
        "study": "M26 predeclared frozen original M20 validation comparison",
        "schema": SCHEMA,
        "original_m20_seed": SEED,
        "n_estimators": TREES,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "bad_prediction_threshold": .5,
        "label_semantics": "Mendeley publisher video-level good/bad, not named form errors",
        "original_train_subjects": len(people["train"]),
        "original_val_subjects": len(people["val"]),
        "original_VAL_pushup_clips": len(rows["val"]),
        "train_full_accepted": len(train_full),
        "train_upper_accepted": len(train_upper),
        "val_full_accepted": len(val_full),
        "val_upper_accepted": len(val_upper),
        "val_upper_only_recovered": len(upper_only),
        "paired_common_val_clips": len(paired),
        "paired_model_scores": {},
        "unpaired_recovered_val_clips": {},
        "original_TEST_video_decoded": False,
        "original_TEST_scored": False,
        "protocol": (
            "Two frozen RF variants fitted only on original M20 TRAIN people; "
            "same original VAL people, same paired common video IDs; "
            "upper-only predictions reported separately. Do NOT tune on VAL."
        ),
    }
    for mode in TRAIN_SETS:
        model = _fit(training[mode], n_trees=TREES)
        validation = val_tables[mode]
        scores = _predict(model, validation)
        truth = validation.label.map({"good": 0, "bad": 1}).to_numpy(dtype=int)
        paired_records = list(zip(truth.tolist(), scores.tolist()))
        report["paired_model_scores"][mode] = {
            "all_common": _metrics(paired_records),
            "by_view": {
                view: _metrics([
                    item for item, original_id in zip(paired_records, paired)
                    if val_full.at[original_id, "view"] == view
                ])
                for view in ("front", "side", "diagonal")
            }
        }
        if mode == "upper_expanded":
            extra = val_upper.loc[upper_only] if upper_only else None
            extra_scores = _predict(model, extra) if extra is not None else []
            extra_metrics = list(zip(
                extra.label.map({"good": 0, "bad": 1}).tolist()
                if extra is not None else [],
                extra_scores.tolist() if extra is not None else [],
            ))
            report["unpaired_recovered_val_clips"] = {
                "all_views": _metrics(extra_metrics),
                "note": "Only upper-body is capable of scoring these clips.",
            }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="Decode original VAL video only")
    extract.add_argument("--zip", type=Path, default=Path(
        "data/external/mendeley_gym7/Dataset Exercise-wise.zip"))
    extract.add_argument("--manifest", type=Path, default=Path(
        "data/external/mendeley_gym7/m20_manifest.csv"))
    extract.add_argument("--original-features", type=Path, default=Path(
        "data/features/m20_gym7/m20_clip_features.csv"))
    extract.add_argument("--output-dir", type=Path, default=Path(
        "data/features/m26_pushup_val_upper_v1"))

    evaluate = commands.add_parser("evaluate", help="One fixed validation comparison")
    evaluate.add_argument("--original-features", type=Path, default=Path(
        "data/features/m20_gym7/m20_clip_features.csv"))
    evaluate.add_argument("--manifest", type=Path, default=Path(
        "data/external/mendeley_gym7/m20_manifest.csv"))
    evaluate.add_argument("--train-dir", type=Path, default=Path(
        "data/features/m23_pushup_train_upper_v1"))
    evaluate.add_argument("--val-dir", type=Path, default=Path(
        "data/features/m26_pushup_val_upper_v1"))
    evaluate.add_argument("--report", type=Path, default=Path(
        "data/processed/m26_pushup_frozen_validation_report.json"))
    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            outcome = extract_validation(
                args.zip, args.manifest, args.original_features, args.output_dir
            )
        else:
            if args.report.exists():
                raise FileExistsError("Validation report exists; refuse overwrite")
            outcome = evaluate_validation(
                args.original_features, args.manifest,
                args.train_dir / "train_full_body.csv",
                args.train_dir / "train_upper_body.csv",
                args.val_dir / "val_full_body.csv",
                args.val_dir / "val_upper_body.csv"
            )
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
            )
    except (ValueError, AssertionError, OSError, RuntimeError, ImportError,
            zipfile.BadZipFile) as error:
        print(f"M26 {args.command} stopped: {error}", file=sys.stderr)
        return 1
    print(json.dumps(outcome, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
