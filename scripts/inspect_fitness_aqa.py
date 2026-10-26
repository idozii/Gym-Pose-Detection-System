#!/usr/bin/env python3
"""Inspect the Fitness-AQA squat labeled archive without unzipping large videos.

Requires only the standard library. This is dataset validation, not model training.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"}
VIDEO_KEY = re.compile(r"^[0-9]+_[0-9]+$")


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; download its small JSON file first")
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def inspect(archive: Path, annotations: Path) -> dict:
    train = _read_json(annotations / "train_keys.json")
    val = _read_json(annotations / "val_keys.json")
    test = _read_json(annotations / "test_keys.json")
    forward = _read_json(annotations / "error_knees_forward.json")
    inward = _read_json(annotations / "error_knees_inward.json")

    if any(not isinstance(items, list) or any(not isinstance(k, str) for k in items)
           for items in (train, val, test)):
        raise ValueError("Each split file must be a list of string video keys")
    if not isinstance(forward, dict) or not isinstance(inward, dict):
        raise ValueError("Error files must map video keys to time intervals")
    splits = {"train": train, "val": val, "test": test}
    for name, keys in splits.items():
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate video IDs within {name} split")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(splits[a]) & set(splits[b])
        if overlap:
            raise ValueError(f"Data leakage: {len(overlap)} shared IDs in {a}/{b}")
    all_keys = set(train) | set(val) | set(test)
    if not all_keys <= set(forward) or not all_keys <= set(inward):
        raise ValueError("Missing some split keys in the error annotation JSON files")
    if any(not VIDEO_KEY.fullmatch(k) for k in all_keys):
        raise ValueError("Unexpected video key syntax; verify dataset version")

    for kind, labels in (("knees_forward", forward), ("knees_inward", inward)):
        for key in all_keys:
            intervals = labels[key]
            if not isinstance(intervals, list):
                raise ValueError(f"{kind}[{key}] must be a list")
            for window in intervals:
                if (not isinstance(window, list) or len(window) != 2
                    or not all(isinstance(x, (int, float)) for x in window)
                    or window[0] < 0 or window[1] < window[0]):
                    raise ValueError(f"Invalid {kind} interval for video {key}: {window!r}")

    with zipfile.ZipFile(archive) as zipfile_handle:
        entries = [i for i in zipfile_handle.infolist() if not i.is_dir()]
        video_files = [i.filename for i in entries
                       if Path(i.filename).suffix.lower() in VIDEO_EXTENSIONS]
        counts = Counter(Path(f).stem for f in video_files)
        duplicates = sorted(k for k, n in counts.items() if n > 1)
        found_keys = set(counts)
        missing_videos = sorted(all_keys - found_keys)
        extra_videos = sorted(found_keys - all_keys)
        archive_bytes = sum(i.file_size for i in entries)

    summary = {
        "split_counts": {name: len(keys) for name, keys in splits.items()},
        "total_unique_split_keys": len(all_keys),
        "annotation_video_counts": {
            "knees_forward": sum(bool(forward[k]) for k in all_keys),
            "knees_inward": sum(bool(inward[k]) for k in all_keys),
        },
        "archive_total_entries": len(entries),
        "archive_uncompressed_bytes": archive_bytes,
        "video_files_found": len(video_files),
        "matched_video_keys": len(all_keys & found_keys),
        "missing_video_keys_count": len(missing_videos),
        "missing_video_keys_examples": missing_videos[:12],
        "extra_video_keys_count": len(extra_videos),
        "extra_video_keys_examples": extra_videos[:12],
        "duplicate_video_stems_count": len(duplicates),
        "duplicate_video_stems_examples": duplicates[:12],
        "first_video_paths": video_files[:10],
        "note": "Dataset-provided splits checked by clip ID, not by subject identity",
    }
    if not video_files:
        raise ValueError("Archive contains no recognized video extensions; inspect ZIP file layout")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="archive", type=Path,
                        default=Path("data/external/fitness_aqa/squat/labeled/videos.zip"))
    parser.add_argument("--annotations", type=Path,
                        default=Path("data/external/fitness_aqa/squat/labeled/annotations"))
    args = parser.parse_args()
    try:
        result = inspect(args.archive, args.annotations)
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(f"FITNESS-AQA CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    if result["missing_video_keys_count"] or result["duplicate_video_stems_count"]:
        print("CHECK INCOMPLETE: archive and labels do not match one-to-one", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
