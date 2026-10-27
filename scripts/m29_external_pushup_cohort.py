"""M29: validate a NEW consented Push-up cohort before independent evaluation.

This tool does not train, score, decode, display or upload source videos.
Only project-local manifest + consent references are inspected. It cannot
prove that pseudonymous volunteers are different from Mendeley people:
the owner must document real-world independence separately.

All cohort records remain in Git-ignored data/external. Reports contain
aggregate counts only, no names, IDs, paths, videos or content hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

SCHEMA_VERSION = "gym7-external-pushup-v1"
COLUMNS = (
    "video_id", "subject_id", "view", "source_file",
    "reviewer_a_id", "reviewer_a_label",
    "reviewer_b_id", "reviewer_b_label",
    "final_label", "adjudicator_id",
    "consent_ref", "age_18plus_verified", "annotation_version",
)
LABELS = {"good", "bad"}
VIEWS = {"front", "side", "diagonal"}
EXT = {".mp4", ".mov", ".mkv", ".avi"}
SAFE_ID = re.compile(r"^[a-zA-Z0-9_-]{4,64}$")
VIDEO_ID = re.compile(r"^extclip_[a-zA-Z0-9_-]{8,48}$")
PERSON_ID = re.compile(r"^extperson_[a-zA-Z0-9_-]{8,48}$")
MAX_CLIP_BYTES = 500 * 1024 * 1024


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_path(root: Path, relative: str) -> Path:
    if "\\" in relative or not relative or len(relative) > 250:
        raise ValueError("Unsafe or missing relative source_file")
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or "." in candidate.parts or ".." in candidate.parts:
        raise ValueError("source_file must be a safe relative path below dataset root")
    if candidate.suffix.lower() not in EXT:
        raise ValueError("Unsupported source video extension")
    path = root.joinpath(*candidate.parts)
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (ValueError, OSError, RuntimeError) as error:
        raise ValueError("Video absent or symlink escapes private dataset root") from error
    if not path.is_file():
        raise ValueError("source_file is not a regular video file")
    if path.stat().st_size == 0 or path.stat().st_size > MAX_CLIP_BYTES:
        raise ValueError("Source clip empty or exceeds 500 MiB")
    return path


def init_manifest(path: Path) -> dict:
    if path.exists():
        raise FileExistsError("M29 source manifest already exists; cannot overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(COLUMNS)
    return {
        "manifest_created": str(path),
        "rows": 0,
        "note": "Fill offline with pseudonymous IDs only after consented recording "
                "and independent video-level annotations; do not commit any CSV."
    }


def audit_cohort(manifest: Path, dataset_root: Path) -> dict:
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("Manifest columns do not match frozen M29 schema")
        rows = list(reader)
    if not rows:
        raise ValueError("No external annotated videos; audit is not yet possible")
    if len(rows) > 10000:
        raise ValueError("Too many videos for a single M29 cohort")

    video_ids, source_paths, sha_set, people = set(), set(), set(), set()
    distribution, agreement, adjudication = Counter(), 0, 0
    for row in rows:
        if None in row or any(x is None or x != x.strip() for x in row.values()):
            raise ValueError("Malformed or untrimmed source manifest values")
        if not VIDEO_ID.fullmatch(row["video_id"]):
            raise ValueError("Use generated-style pseudonymous extclip_ video IDs")
        if not PERSON_ID.fullmatch(row["subject_id"]):
            raise ValueError("Use separate extperson_ IDs for genuinely NEW participants")
        if row["video_id"] in video_ids:
            raise ValueError("Duplicate video_id; independent observations required")
        video_ids.add(row["video_id"])
        people.add(row["subject_id"])
        if row["source_file"] in source_paths:
            raise ValueError("Duplicate source_file in external cohort manifest")
        source_paths.add(row["source_file"])

        if row["view"] not in VIEWS or row["final_label"] not in LABELS:
            raise ValueError("Invalid original view or external good/bad label")
        if row["reviewer_a_label"] not in LABELS or row["reviewer_b_label"] not in LABELS:
            raise ValueError("Both reviewers must assign a good/bad video-level label")
        ra, rb, adjudicator = (
            row["reviewer_a_id"], row["reviewer_b_id"], row["adjudicator_id"]
        )
        if (not SAFE_ID.fullmatch(ra) or not SAFE_ID.fullmatch(rb) or ra == rb):
            raise ValueError("Two distinct pseudonymous independent reviewer IDs required")
        if row["reviewer_a_label"] == row["reviewer_b_label"]:
            if row["final_label"] != row["reviewer_a_label"] or adjudicator:
                raise ValueError("Agreeing reviewer labels must match final label, no adjudicator")
            agreement += 1
        else:
            if (not SAFE_ID.fullmatch(adjudicator)
                or adjudicator in {ra, rb}):
                raise ValueError("Disagreement requires a third independent adjudicator")
            adjudication += 1
        if (not SAFE_ID.fullmatch(row["consent_ref"]) or
            row["age_18plus_verified"] != "yes"):
            raise ValueError("Each adult volunteer needs a valid private consent reference")
        if row["annotation_version"] != SCHEMA_VERSION:
            raise ValueError("Incorrect annotation rubric version")

        path = _video_path(dataset_root, row["source_file"])
        digest = _file_hash(path)
        if digest in sha_set:
            raise ValueError("Duplicate video bytes - cannot count same clip twice")
        sha_set.add(digest)
        distribution[(row["view"], row["final_label"])] += 1

    n = len(rows)
    report = {
        "audit": "M29 independent Push-up annotation and source-manifest QA",
        "annotation_schema": SCHEMA_VERSION,
        "annotated_clips": n,
        "distinct_pseudonymous_participants": len(people),
        "two_reviewer_agreements": agreement,
        "third_reviewer_adjudications": adjudication,
        "agreement_fraction": agreement / n,
        "by_view_and_label": {
            f"{view}/{label}": distribution[(view,label)]
            for view in sorted(VIEWS) for label in sorted(LABELS)
        },
        "all_consent_refs_present": True,
        "all_adult_verifications_recorded": True,
        "all_files_exist_and_are_byte_distinct": True,
        "ready_for_pilot": (len(people) >= 5 and
                            all(distribution[(view,label)] for view in VIEWS
                                for label in LABELS)),
        "ready_for_large_independent_study": (
            len(people) >= 20 and n >= 120 and
            min(distribution.values()) >= 10 and
            all(distribution[(view,label)] >= 10
                for view in VIEWS for label in LABELS)
        ),
        "external_source_subject_independence_verified": False,
        "model_predictions_or_labels_from_original_TEST_accessed": False,
        "warning": (
            "Pseudonymous IDs and signatures cannot prove participants are "
            "independent of Gym7. The owner must document recruitment "
            "provenance and independent human labeling before any test. "
            "Pilot/study readiness criteria are planning targets, NOT proof "
            "of model efficacy or clinical correctness."
        )
    }
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    cmd=parser.add_subparsers(dest="command",required=True)
    init=cmd.add_parser("init",help="Create a blank private CSV header")
    init.add_argument("--manifest",type=Path,default=Path(
        "data/external/m29_pushup/manifest.csv"))
    audit=cmd.add_parser("audit",help="Check consented source cohort before evaluation")
    audit.add_argument("--manifest",type=Path,default=Path(
        "data/external/m29_pushup/manifest.csv"))
    audit.add_argument("--root",type=Path,default=Path(
        "data/external/m29_pushup/videos"))
    audit.add_argument("--report",type=Path,default=Path(
        "data/processed/m29_external_pushup_source_audit.json"))
    args=parser.parse_args(argv)
    try:
        if args.command=="init":
            result=init_manifest(args.manifest)
        else:
            if args.report.exists():
                raise FileExistsError("M29 source audit report exists; refuse overwrite")
            result=audit_cohort(args.manifest,args.root)
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    except (OSError,ValueError,RuntimeError) as error:
        print(f"M29 {args.command} failed: {error}",file=sys.stderr)
        return 1
    print(json.dumps(result,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
