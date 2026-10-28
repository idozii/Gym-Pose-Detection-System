"""M54: TRAIN-only RepCount-A annotation semantics and origin evidence audit.

Public author sources document that RepCount-A videos were originally
collected from YouTube and annotated with per-period start/end positions.
That DOES NOT prove specific cycle phase semantics, legal re-use rights,
identity independence, or provenance of the owner's tar.gz bytes.

This local audit scans ONLY the embedded TRAIN CSV (no MP4 extraction,
no TEST/VALID annotation requests) and verifies M49's original 17-source
Git/source lock, archive TRAIN CSV SHA and the exact original M50
20+20 filename-hash cohorts. It publishes only aggregate counts and
schema names, no raw identifiers, labels, filenames or clip annotations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

from scripts.m37_repcount_archive import parse_official_csv
from scripts.m37_repcount_tar import _embedded_csv
from scripts.m49_pretest_git_provenance import run as verify_m49
from scripts.m50_train_label_video_integrity import (
    _publisher_cohorts, _header_metadata
)

PROTOCOL = "m54-repcount-train-origin-annotation-semantics-audit-v1"
ROOT = Path("data/processed/m54_train_annotation_provenance.json")
PUBLIC_AUTHOR_SOURCES = {
    "official_dataset": "https://svip-lab.github.io/dataset/RepCount_dataset.html",
    "official_repo": "https://github.com/SvipRepetitionCounting/TransRAC"
}
KNOWN_UNKNOWN = (
    "publisher_archive_byte_authenticity",
    "source_video_redistribution_license",
    "commercial_use_permission",
    "human_start_end_phase_alignment",
    "participant_session_camera_independence"
)
# M37's original parser validates the required author schema and L1..Ln.
# Optional CSV export/index columns may contain punctuation or blank headers.
# Never publish their raw header names or data; summarize only their count.
LABEL_COLUMN = re.compile(r"L[1-9]\d*")


def audit_train_csv(raw: bytes) -> dict:
    """Only schema and aggregate positive/zero interval properties."""
    rows = parse_official_csv(raw)
    groups = _publisher_cohorts(raw)
    if [len(g) for g in groups] != [20,20] or [
        sum(item["annotated_count"] for item in g) for g in groups
    ] != [312,209]:
        raise ValueError("M54 expected the unchanged M46 TRAIN cohort 20+20 / GT521")

    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    fields = reader.fieldnames or []
    # The original M37 TRAIN parser already succeeded above, validating
    # the required label sequence and enforcing a 20-MiB CSV byte ceiling.
    # RepCount files may contain hundreds or thousands of L1..Ln columns;
    # an arbitrary M54-specific 300-header cap rejects original signed CSV.
    if not fields:
        raise ValueError("M54 publisher TRAIN CSV has no header")
    # Do not enforce an ASCII identifier regex on OTHER columns. The frozen
    # M37 parser already accepted the signed TRAIN CSV, and real CSV exporters
    # can include unnamed/index/annotation auxiliary columns. Only the
    # original validated label contract is relevant to counting.
    known = [name for name in fields if isinstance(name, str) and (
        name in {"name", "type", "count"} or
        LABEL_COLUMN.fullmatch(name) is not None
    )]
    if (not {"name", "type", "count", "L1", "L2"}.issubset(known) or
        len(set(known)) != len(known)):
        raise ValueError("M54 publisher TRAIN required label columns ambiguous")
    # Redact raw optional column names: they could contain user metadata.
    # Never write them, raw row values or source filenames to JSON output.
    auxiliary_header_count = len(fields) - len(known)
    metadata = _header_metadata(raw, [[r["video"].lower() for r in g]
                                       for g in groups])
    positive = [x for x in rows if x["annotated_count"]>0]
    zero = [x for x in rows if x["annotated_count"]==0]
    summary = {}
    for label, group in zip(("development_20","replication_20"),groups):
        count = len(group)
        gt = sum(x["annotated_count"] for x in group)
        lengths = []
        gaps = []
        starts_at_frame_zero = 0
        for item in group:
            bounds = item["cycle_bounds_frames"]
            starts_at_frame_zero += bool(bounds and bounds[0] == 0)
            for k in range(0,len(bounds),2):
                lo,hi = bounds[k:k+2]
                lengths.append(hi-lo)
                if k: gaps.append(lo-bounds[k-1])
        if len(lengths)!=gt or any(v<=0 for v in lengths) or any(v<0 for v in gaps):
            raise ValueError("M54 cycle pairs inconsistent with original parser")
        summary[label] = {
            "source_files": count,
            "annotated_repetitions": gt,
            "positive_gt_videos": sum(x["annotated_count"]>0 for x in group),
            "zero_gt_videos": sum(x["annotated_count"]==0 for x in group),
            "total_valid_positive_frame_spans": len(lengths),
            "nonoverlapping_adjacent_cycle_gaps": len(gaps),
            "adjacent_cycle_gaps_with_zero_unlabeled_frames":
                sum(v==0 for v in gaps),
            "adjacent_cycle_gaps_with_positive_unlabeled_frames":
                sum(v>0 for v in gaps),
            "first_cycle_starts_at_frame_zero": starts_at_frame_zero,
            "smallest_cycle_span_frames": min(lengths) if lengths else None,
            "largest_cycle_span_frames": max(lengths) if lengths else None,
            "frame_boundary_semantics_confirmed_by_authors": False
        }

    return {
        "train_squat_annotation_rows":len(rows),
        "train_squat_positive_gt_videos":len(positive),
        "train_squat_zero_gt_videos":len(zero),
        "publisher_train_csv_column_names":known,
        "publisher_train_csv_auxiliary_column_count":auxiliary_header_count,
        "publisher_train_csv_auxiliary_column_names_persisted":False,
        "recognized_explicit_person_session_camera_schema":metadata,
        "original_frozen_cohorts":summary,
        "count_matches_pairs_under_original_parser":True,
        "frame_bounds_are_ordered_nonoverlapping_for_squat_train":True,
        "gt_frame_position_is_not_an_anatomic_phase_label":True,
        "original_current_interval_parser_treats_bounds_as_half_open":True,
        "original_half_open_interpretation_verified_against_human_annotator_policy":False
    }


def run(*, archive:Path, report:Path, m46:Path, m47:Path,
        lock:Path, csv_reader=None, verifier=None) -> dict:
    if report.exists():
        raise FileExistsError("M54 refuses overwriting an existing result")
    gate = (verifier or verify_m49)(
        m46_path=m46,m47_path=m47,lock_path=lock,
        report_path=Path("data/processed/m49_pretest_provenance.json"),
        check_only=True
    )
    if (gate.get("research_source_provenance_check_passed") is not True or
        gate.get("official_test_executed_or_opened") is not False):
        raise ValueError("M54 original M49 source/Git gate not verified")
    m48=json.loads(lock.read_bytes())
    if m48.get("source_split") != "train":
        raise ValueError("M54 requires M48 TRAIN-only lock")
    raw=(csv_reader or _embedded_csv)(archive,"train")
    sha=hashlib.sha256(raw).hexdigest()
    if sha!=m48.get("publisher_train_annotation_sha256"):
        raise ValueError("M54 original archive TRAIN CSV SHA differs from M48")
    source=audit_train_csv(raw)
    report_payload={
        "protocol":PROTOCOL,
        "source_split":"train",
        "original_train_annotation_sha256":sha,
        "owner_tar_full_sha256_calculated":False,
        "official_publisher_archive_byte_parity_verified":False,
        "original_m49_research_source_gate_passed":True,
        "official_publisher_sources":PUBLIC_AUTHOR_SOURCES,
        "public_source_summary":{
            "part_a_origin_described_by_authors":"YouTube",
            "annotation_description":"count and cycle location on time axis",
            "declared_original_part_a_video_count":1041,
            "full_archive_publisher_sha256_published_or_checked":False,
            "license_authority_independently_confirmed":False
        },
        "train_label_aggregate":source,
        "license_and_provenance_blockers":{key:"unverified" for key in KNOWN_UNKNOWN},
        "test_or_validation_csv_or_media_opened":False,
        "video_bytes_extracted_or_decoded":False,
        "filename_ids_or_cycle_boundaries_saved":False,
        "research_thresholds_or_event_matching_changed":False
    }
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open("x",encoding="utf8") as target:
        json.dump(report_payload,target,indent=2,allow_nan=False)
        target.write("\n")
    return report_payload


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive",required=True,type=Path)
    parser.add_argument("--m46-report",type=Path,
        default=Path("data/processed/m46_train_frozen_replication.json"))
    parser.add_argument("--m47-report",type=Path,
        default=Path("data/processed/m47_train_replication_bootstrap.json"))
    parser.add_argument("--m48-lock",type=Path,
        default=Path("data/processed/m48_train_evidence_freeze.json"))
    parser.add_argument("--report",type=Path,default=ROOT)
    args=parser.parse_args(argv)
    try:
        result=run(archive=args.source_archive,report=args.report,
                   m46=args.m46_report,m47=args.m47_report,lock=args.m48_lock)
    except (OSError,ValueError,TypeError,KeyError,RuntimeError,
            UnicodeError,json.JSONDecodeError) as error:
        print("M54 refused: "+str(error),file=sys.stderr)
        return 1
    print(json.dumps({
        "protocol":result["protocol"],
        "source_split":result["source_split"],
        "train_squat_rows":result["train_label_aggregate"]["train_squat_annotation_rows"],
        "train_zero_rep_clips":result["train_label_aggregate"]["train_squat_zero_gt_videos"],
        "cohorts":result["train_label_aggregate"]["original_frozen_cohorts"],
        "known_unknowns":result["license_and_provenance_blockers"],
        "official_TEST_accessed":False,
        "report_path":str(args.report)
    },indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
