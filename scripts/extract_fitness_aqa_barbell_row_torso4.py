#!/usr/bin/env python3
"""Barbell Row shoulder+hip-only 20-feature image extraction, TRAIN/VAL only.

Uses the source-tested Barbell Row image pipeline with a distinct profile
and default output directory. Will not overwrite original 32-feature CSVs.
Publisher TEST photo pixels remain locked.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

from scripts.extract_fitness_aqa_barbell_row import extract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path(
        "data/external/fitness_aqa/barbell_row/labeled/images.zip"))
    parser.add_argument("--annotations", type=Path, default=Path(
        "data/external/fitness_aqa/barbell_row/labeled/annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/features/fitness_aqa_barbell_row_torso4"))
    parser.add_argument("--split", choices=("train", "val", "train-val"),
                        default="train-val")
    parser.add_argument("--max-images-per-task-split", type=int, default=12,
                        help="Use 0 for full publisher TRAIN+VAL per target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-visibility", type=float, default=.5)
    parser.add_argument("--max-image-mb", type=int, default=24)
    parser.add_argument("--max-pixels", type=int, default=30_000_000)
    args = parser.parse_args()
    splits = ("train", "val") if args.split == "train-val" else (args.split,)
    try:
        report = extract(
            args.zip, args.annotations, args.output_dir,
            selected_splits=splits, feature_profile="torso4",
            max_images_per_task_split=args.max_images_per_task_split,
            seed=args.seed, min_visibility=args.min_visibility,
            max_image_mb=args.max_image_mb, max_pixels=args.max_pixels)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            ImportError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        parser.exit(1, f"Barbell Row torso4 extraction failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 2 if any(
        stat["image_decode_failures"] for stat in report["splits"].values()
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
