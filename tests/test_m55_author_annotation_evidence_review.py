"""M55 deterministic TRAIN aggregate and author-evidence gate (no archive/TEST)."""
import copy
import json
from pathlib import Path

import pytest

from scripts import m55_author_annotation_evidence_review as m55


def owner_m54_fixture():
    def cohort(gt,gap0,gap1,frame0=0,max_span=158):
        return {
            "source_files":20,
            "annotated_repetitions":gt,
            "positive_gt_videos":20,
            "zero_gt_videos":0,
            "total_valid_positive_frame_spans":gt,
            "nonoverlapping_adjacent_cycle_gaps":gt-20,
            "adjacent_cycle_gaps_with_zero_unlabeled_frames":gap0,
            "adjacent_cycle_gaps_with_positive_unlabeled_frames":gap1,
            "first_cycle_starts_at_frame_zero":frame0,
            "smallest_cycle_span_frames":24,
            "largest_cycle_span_frames":max_span,
            "frame_boundary_semantics_confirmed_by_authors":False
        }
    return {
        "protocol":m55.M54_PROTOCOL,
        "source_split":"train",
        "original_m49_research_source_gate_passed":True,
        "test_or_validation_csv_or_media_opened":False,
        "video_bytes_extracted_or_decoded":False,
        "filename_ids_or_cycle_boundaries_saved":False,
        "original_train_annotation_sha256":"a"*64,
        "license_and_provenance_blockers":{
            name:"unverified" for name in m55.BLOCKERS
        },
        "train_label_aggregate":{
            "train_squat_annotation_rows":101,
            "train_squat_zero_gt_videos":0,
            "count_matches_pairs_under_original_parser":True,
            "frame_bounds_are_ordered_nonoverlapping_for_squat_train":True,
            "gt_frame_position_is_not_an_anatomic_phase_label":True,
            "original_half_open_interpretation_verified_against_human_annotator_policy":False,
            "publisher_train_csv_auxiliary_column_names_persisted":False,
            "original_frozen_cohorts":{
                "development_20":cohort(312,157,135),
                "replication_20":cohort(209,155,34,1,166)
            }
        }
    }


def lock_fixture():
    return {"source_split":"train",
            "publisher_train_annotation_sha256":"a"*64}


def test_m55_matches_owner_real_m54_counts_with_no_test():
    report=m55.review(owner_m54_fixture(),lock_fixture())
    assert report["protocol"]==m55.PROTOCOL
    assert report["source_split"]=="train"
    assert report["owner_original_train_observations"]["squat_source_videos"]==101
    assert report["owner_original_train_observations"]["zero_GT_squat_videos"]==0
    assert report["owner_original_train_observations"]["selected_GT_cycles"]==521
    cohorts=report["owner_original_train_observations"]["selected_cohorts"]
    assert cohorts["development_20"]["adjacent_zero_gap_pairs"]==157
    assert cohorts["replication_20"]["adjacent_positive_gap_pairs"]==34
    assert cohorts["replication_20"]["first_cycle_starts_at_frame_zero"]==1
    assert cohorts["replication_20"]["max_cycle_length_frames"]==166
    assert all(value is False for key,value in report["claim_gate"].items()
               if key.startswith("can_claim"))
    assert report["claim_gate"]["can_report_m54_train_aggregate_consistency"]
    assert report["official_test_or_validation_opened"] is False
    assert report["author_contact_or_permission_claimed"] is False
    assert "L1" not in json.dumps(report)
    assert "source_video_filename" not in json.dumps(report)


@pytest.mark.parametrize("change", [
    lambda x:x.update(source_split="test"),
    lambda x:x.update(original_m49_research_source_gate_passed=False),
    lambda x:x.update(test_or_validation_csv_or_media_opened=True),
    lambda x:x.update(video_bytes_extracted_or_decoded=True),
    lambda x:x["train_label_aggregate"].update(train_squat_annotation_rows=40),
    lambda x:x["train_label_aggregate"].update(train_squat_zero_gt_videos=1),
    lambda x:x["train_label_aggregate"].update(
        original_half_open_interpretation_verified_against_human_annotator_policy=True),
    lambda x:x["train_label_aggregate"]["original_frozen_cohorts"][
        "development_20"].update(annotated_repetitions=313),
    lambda x:x["train_label_aggregate"]["original_frozen_cohorts"][
        "replication_20"].update(adjacent_cycle_gaps_with_positive_unlabeled_frames=35),
    lambda x:x["license_and_provenance_blockers"].update(
        commercial_use_permission="verified"),
    lambda x:x["train_label_aggregate"].update(
        publisher_train_csv_auxiliary_column_names_persisted=True),
])
def test_m55_refuses_changing_original_safety_and_claims(change):
    data=owner_m54_fixture()
    change(data)
    with pytest.raises(ValueError):
        m55.review(data,lock_fixture())


def test_m55_refuses_wrong_train_sha_or_missing_lock():
    data=owner_m54_fixture()
    lock=lock_fixture()
    lock["publisher_train_annotation_sha256"]="b"*64
    with pytest.raises(ValueError,match="SHA"):
        m55.review(data,lock)
    lock["publisher_train_annotation_sha256"]="a"*64
    lock["source_split"]="validation"
    with pytest.raises(ValueError):
        m55.review(data,lock)


def test_m55_outputs_only_aggregate_and_refuses_overwrite(tmp_path):
    m54_path=tmp_path/"m54.json"
    m48_path=tmp_path/"m48.json"
    out=tmp_path/"m55.json"
    m54_path.write_text(json.dumps(owner_m54_fixture()),encoding="utf8")
    m48_path.write_text(json.dumps(lock_fixture()),encoding="utf8")
    result=m55.run(m54_path=m54_path,m48_path=m48_path,report_path=out)
    assert json.loads(out.read_text(encoding="utf8"))==result
    assert result["claim_gate"]["can_claim_publisher_test_performance"] is False
    assert result["interpretation"]["half_open_frame_endpoint_semantics"] == (
        "project_parser_convention_not_author_verified")
    assert "source_videos_filenames_gt_boundaries_persisted" in result
    with pytest.raises(FileExistsError):
        m55.run(m54_path=m54_path,m48_path=m48_path,report_path=out)


def test_m55_documents_only_author_protocol_not_license_or_phase():
    assert m55.AUTHOR_EVIDENCE["cvpr_2022_paper"]["status"]=="documented_by_paper"
    assert m55.AUTHOR_EVIDENCE["dataset_homepage"]["status"]=="documented_by_dataset_authors"
    report=m55.review(owner_m54_fixture(),lock_fixture())
    assert report["claim_gate"]["can_claim_gt_boundaries_match_anatomical_phases"] is False
    assert report["claim_gate"]["can_claim_commercial_or_video_redistribution_license"] is False
    assert report["claim_gate"]["can_claim_subject_disjoint_generalization"] is False
