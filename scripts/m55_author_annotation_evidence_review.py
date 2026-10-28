"""M55 read-only RepCount-A author-evidence and M54 TRAIN annotation gate.

Consumes ONLY owner-local M54 aggregate JSON and frozen M48 lock; no
publisher archive, video, CSV, TEST/VALID data, browser or image access.
M55 is a provenance/semantic claims review, NOT independent licensing
authorization, frame phase labeling, or participant identification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

PROTOCOL = "m55-repcount-author-evidence-and-annotation-gate-v1"
M54_PROTOCOL = "m54-repcount-train-origin-annotation-semantics-audit-v1"
MAX_JSON_BYTES = 4 * 1024 * 1024
SHA256 = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_M54 = Path("data/processed/m54_train_annotation_provenance.json")
DEFAULT_M48 = Path("data/processed/m48_train_evidence_freeze.json")
DEFAULT_REPORT = Path("data/processed/m55_author_evidence_review.json")

AUTHOR_EVIDENCE = {
    "dataset_homepage": {
        "url": "https://svip-lab.github.io/dataset/RepCount_dataset.html",
        "documented": "Part-A comprises 1041 YouTube-sourced videos; labels describe count and temporal cycle location",
        "status": "documented_by_dataset_authors"
    },
    "cvpr_2022_paper": {
        "url": "https://openaccess.thecvf.com/content/CVPR2022/html/Hu_TransRAC_Encoding_Multi-Scale_Temporal_Correlation_With_Transformers_for_Repetitive_Action_CVPR_2022_paper.html",
        "documented": "Each action cycle has a start/end time; two annotators per video and cross-check when total counts differ by more than one",
        "status": "documented_by_paper"
    },
    "transrac_readme": {
        "url": "https://github.com/SvipRepetitionCounting/TransRAC",
        "documented": "Authors distinguish retrained cross-model comparisons and publish an experimental-setting clarification",
        "status": "documented_by_authors"
    }
}

BLOCKERS = (
    "publisher_archive_byte_authenticity",
    "source_video_redistribution_license",
    "commercial_use_permission",
    "human_start_end_phase_alignment",
    "participant_session_camera_independence"
)

def read_private_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("M55 input must be a small regular private JSON")
    content = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(content, dict):
        raise ValueError("M55 expected a JSON object")
    return content


def review(m54: dict, m48: dict) -> dict:
    """Fail closed on changed scope/cohorts. Only aggregate status is emitted."""
    if (m54.get("protocol") != M54_PROTOCOL or
        m54.get("source_split") != "train" or
        m48.get("source_split") != "train" or
        m54.get("original_m49_research_source_gate_passed") is not True or
        m54.get("test_or_validation_csv_or_media_opened") is not False or
        m54.get("video_bytes_extracted_or_decoded") is not False or
        m54.get("filename_ids_or_cycle_boundaries_saved") is not False):
        raise ValueError("M55 requires verified TRAIN-only M54 source gate")
    sha=m54.get("original_train_annotation_sha256")
    if (not isinstance(sha,str) or not SHA256.fullmatch(sha) or
        sha != m48.get("publisher_train_annotation_sha256")):
        raise ValueError("M55 M54 original TRAIN annotation SHA differs from M48")
    x=m54.get("train_label_aggregate")
    if not isinstance(x,dict) or x.get("train_squat_annotation_rows") != 101 or x.get("train_squat_zero_gt_videos") != 0:
        raise ValueError("M55 original 101 positive-GT TRAIN Squat rows not verified")
    if (x.get("count_matches_pairs_under_original_parser") is not True or
        x.get("frame_bounds_are_ordered_nonoverlapping_for_squat_train") is not True or
        x.get("gt_frame_position_is_not_an_anatomic_phase_label") is not True or
        x.get("original_half_open_interpretation_verified_against_human_annotator_policy") is not False):
        raise ValueError("M55 original parser/annotation semantics scope changed")
    if x.get("publisher_train_csv_auxiliary_column_names_persisted") is not False:
        raise ValueError("M55 redaction gate failed")
    c=x.get("original_frozen_cohorts")
    if not isinstance(c,dict) or set(c)!={"development_20","replication_20"}:
        raise ValueError("M55 requires original M46 20+20 hash cohorts")
    results={}
    # Exact owner-run M54 aggregates are part of the frozen TRAIN claim gate.
    # A matching sum alone could hide changed boundary/gap classifications.
    expected = (
        ("development_20", 312, 157, 135, 0, 24, 158),
        ("replication_20", 209, 155, 34, 1, 24, 166),
    )
    for name,n_gt,zero_gaps,positive_gaps,at_zero,min_span,max_span in expected:
        row=c[name]
        n=row.get("source_files")
        if (n != 20 or row.get("annotated_repetitions") != n_gt or
            row.get("total_valid_positive_frame_spans") != n_gt or
            row.get("positive_gt_videos") != 20 or
            row.get("zero_gt_videos") != 0 or
            row.get("frame_boundary_semantics_confirmed_by_authors") is not False or
            row.get("adjacent_cycle_gaps_with_zero_unlabeled_frames") != zero_gaps or
            row.get("adjacent_cycle_gaps_with_positive_unlabeled_frames") != positive_gaps or
            row.get("first_cycle_starts_at_frame_zero") != at_zero or
            row.get("smallest_cycle_span_frames") != min_span or
            row.get("largest_cycle_span_frames") != max_span or
            row.get("nonoverlapping_adjacent_cycle_gaps") != n_gt-n or
            row.get("adjacent_cycle_gaps_with_zero_unlabeled_frames",0)+
            row.get("adjacent_cycle_gaps_with_positive_unlabeled_frames",0) != n_gt-n):
            raise ValueError("M55 original M54 selected cohort interval totals changed")
        results[name]={
            "source_files":20,
            "publisher_GT_cycles":n_gt,
            "pairs_of_adjacent_cycles":n_gt-20,
            "adjacent_zero_gap_pairs":row["adjacent_cycle_gaps_with_zero_unlabeled_frames"],
            "adjacent_positive_gap_pairs":row["adjacent_cycle_gaps_with_positive_unlabeled_frames"],
            "first_cycle_starts_at_frame_zero":row["first_cycle_starts_at_frame_zero"],
            "min_cycle_length_frames":row["smallest_cycle_span_frames"],
            "max_cycle_length_frames":row["largest_cycle_span_frames"]
        }
    blockers=m54.get("license_and_provenance_blockers")
    if (not isinstance(blockers,dict) or set(blockers)!=set(BLOCKERS) or
        any(value!="unverified" for value in blockers.values())):
        raise ValueError("M55 cannot upgrade unknown license or source identities")
    return {
        "protocol":PROTOCOL,
        "scope":"original TRAIN Squat annotation metadata only",
        "source_split":"train",
        "m54_sha256_matches_original_m48_lock":True,
        "author_documented_evidence":AUTHOR_EVIDENCE,
        "owner_original_train_observations":{
            "squat_source_videos":101,
            "zero_GT_squat_videos":0,
            "selected_source_videos":40,
            "selected_GT_cycles":521,
            "selected_cohorts":results,
            "zero_count_false_positive_specificity_unmeasured":True,
            "original_frame_intervals_ordered_and_nonoverlapping":True
        },
        "interpretation":{
            "period_start_and_end":"author_documented_temporal_cycle_intervals",
            "two_annotators_and_discrepancy_review":"paper_described_not_per_clip_audited",
            "anatomical_bottom_or_full_extension":"not_documented_or_independently_verified",
            "half_open_frame_endpoint_semantics":"project_parser_convention_not_author_verified",
            "same_subject_or_session_across_filenames":"unverified",
            "publisher_video_tar_byte_authenticity":"unverified",
            "public_video_redistribution_and_commercial_rights":"unverified"
        },
        "claim_gate":{
            "can_report_m54_train_aggregate_consistency":True,
            "can_report_paper_general_temporal_cycle_protocol":True,
            "can_claim_official_original_archive_verified":False,
            "can_claim_commercial_or_video_redistribution_license":False,
            "can_claim_subject_disjoint_generalization":False,
            "can_claim_gt_boundaries_match_anatomical_phases":False,
            "can_claim_zero_rep_specificity":False,
            "can_claim_publisher_test_performance":False
        },
        "unresolved_evidence_requests":[
            "author_origin_download_link_or_signed_original_archive_digest",
            "explicit_video_creator_rights_and_dataset_use_terms",
            "human_annotation_guideline_for_cycle_start_end_and_frame_index_convention",
            "subject_session_camera_grouping_or_independent_metadata"
        ],
        "source_videos_filenames_gt_boundaries_persisted":False,
        "official_test_or_validation_opened":False,
        "author_contact_or_permission_claimed":False
    }


def run(*,m54_path:Path,m48_path:Path,report_path:Path) -> dict:
    if report_path.exists():
        raise FileExistsError("M55 refuses to overwrite an existing review")
    review_report=review(read_private_json(m54_path),read_private_json(m48_path))
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open("x",encoding="utf8") as output:
        json.dump(review_report,output,indent=2,allow_nan=False)
        output.write("\n")
    return review_report


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m54-report",type=Path,default=DEFAULT_M54)
    parser.add_argument("--m48-lock",type=Path,default=DEFAULT_M48)
    parser.add_argument("--report",type=Path,default=DEFAULT_REPORT)
    args=parser.parse_args(argv)
    try:
        result=run(m54_path=args.m54_report,m48_path=args.m48_lock,
                   report_path=args.report)
    except (OSError,ValueError,TypeError,KeyError,
            json.JSONDecodeError) as error:
        print("M55 refused: "+str(error),file=sys.stderr)
        return 1
    print(json.dumps({
        "protocol":result["protocol"],
        "train_only":result["source_split"]=="train",
        "selected_cycles":result["owner_original_train_observations"]["selected_GT_cycles"],
        "squat_train_videos":result["owner_original_train_observations"]["squat_source_videos"],
        "zero_gt_squat_videos":result["owner_original_train_observations"]["zero_GT_squat_videos"],
        "can_claim_publisher_archive_authenticity":result["claim_gate"]["can_claim_official_original_archive_verified"],
        "can_claim_annotated_phase_alignment":result["claim_gate"]["can_claim_gt_boundaries_match_anatomical_phases"],
        "unresolved_items":result["unresolved_evidence_requests"],
        "test_or_validation_opened":False,
        "report_path":str(args.report)
    },indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
