"""M56 synthetic TRAIN-only preregistration tests. No source TAR or TEST."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import m48_freeze_repcount_evidence as m48
from scripts import m49_pretest_git_provenance as m49
from scripts import m55_author_annotation_evidence_review as m55
from scripts import m56_preregister_evaluation_readiness as m56


def fixtures():
    sha_train="a"*64
    sha_lock="b"*64
    sha_m55="c"*64
    m48_lock={
        "protocol":m48.PROTOCOL,"source_split":"train",
        "owner_observed_evidence_locked":True,
        "publisher_train_annotation_sha256":sha_train,
        "source_code_sha256":{x:"d"*64 for x in m48.SOURCES},
        "known_protocol":{
            "selected_method":"relative_excursion","control_method":"low_to_high",
            "source_split":"train",
            "primary_diagnostic_cadence":"dense_200ms",
            "secondary_cadence":"subsample_400ms",
            "original_gt_event_matcher":"m37_one_to_one_gt_end_tolerance",
            "cohort_video_count":20,"ground_truth_reps":209,
            "bootstrap_unit":"source_video","bootstrap_draws":5000,
            "bootstrap_seed":2026,
            "single_predeclared_test_execution_permitted_by_this_script":False
        }
    }
    m55_report={
        "protocol":m55.PROTOCOL,"source_split":"train",
        "m54_sha256_matches_original_m48_lock":True,
        "source_videos_filenames_gt_boundaries_persisted":False,
        "official_test_or_validation_opened":False,
        "author_contact_or_permission_claimed":False,
        "owner_original_train_observations":{
            "squat_source_videos":101,"zero_GT_squat_videos":0,
            "selected_source_videos":40,"selected_GT_cycles":521,
            "zero_count_false_positive_specificity_unmeasured":True
        },
        "interpretation":{
            "half_open_frame_endpoint_semantics":"project_parser_convention_not_author_verified",
            "same_subject_or_session_across_filenames":"unverified"
        },
        "claim_gate":{
            "can_report_m54_train_aggregate_consistency":True,
            "can_claim_official_original_archive_verified":False,
            "can_claim_commercial_or_video_redistribution_license":False,
            "can_claim_subject_disjoint_generalization":False,
            "can_claim_gt_boundaries_match_anatomical_phases":False,
            "can_claim_zero_rep_specificity":False,
            "can_claim_publisher_test_performance":False
        }
    }
    check={
        "protocol":m49.PROTOCOL,"source_split_used_by_m49":"train",
        "original_m48_lock_sha256":sha_lock,
        "research_source_provenance_check_passed":True,
        "frozen_training_annotation_sha256":sha_train,
        "source_git_provenance":{
            "frozen_source_paths_checked":17,
            "source_bytes_match_original_m48_git_commit":True,
            "source_bytes_match_current_git_commit":True,
            "local_source_bytes_match_m48_lock":True
        },
        "external_test_readiness_approved":False,
        "official_test_executed_or_opened":False,
        "validation_or_test_media_scanned":False,
        "no_original_counter_modified":True,
        "preregistered_future_test_protocol_without_execution":{
            "test_execution_authorized_by_this_tool":False,
            "future_no_test_adaptation":True,"future_test_runs_planned":1
        }
    }
    return m55_report,m48_lock,check,sha_m55,sha_lock


def review(m55_report=None,lock=None,check=None):
    a,b,c,s,t=fixtures()
    return m56.preregister(a if m55_report is None else m55_report,
        b if lock is None else lock,c if check is None else check,
        m55_sha=s,m48_sha=t)

def test_m56_never_authorizes_test_and_preregisters_metrics():
    report=review()
    assert report["protocol"]==m56.PROTOCOL
    assert report["input_split"]=="train"
    assert report["status"].startswith("BLOCKED")
    assert report["test_execution_authorized"] is False
    assert report["publisher_test_accessed"] is False
    assert report["publisher_validation_accessed"] is False
    assert report["test_manifest_inspected_or_saved"] is False
    assert set(report["manual_external_blockers"])==set(m56.BLOCKERS)
    proposal=report["future_study_preregistration_proposal"]
    assert proposal["test_cohort_manifest_status"]=="NOT_DEFINED_OR_ACCESSED"
    assert "750ms" in proposal["original_m37_matching_window"]
    assert proposal["primary_condition"].startswith("dense_200ms")
    assert "GT=0" in proposal["negative_gt_policy"]
    assert proposal["uncertainty"].startswith("predeclared 5000-draw")
    assert report["data_media_decoded_or_extracted"] is False
    assert "filename" not in json.dumps(report).lower()

@pytest.mark.parametrize("mutator",[
    lambda a,b,c:a.update(source_split="test"),
    lambda a,b,c:a.update(official_test_or_validation_opened=True),
    lambda a,b,c:a["claim_gate"].update(can_claim_publisher_test_performance=True),
    lambda a,b,c:a["claim_gate"].update(can_claim_subject_disjoint_generalization=True),
    lambda a,b,c:a["claim_gate"].update(can_claim_gt_boundaries_match_anatomical_phases=True),
    lambda a,b,c:a["owner_original_train_observations"].update(zero_GT_squat_videos=1),
    lambda a,b,c:a["owner_original_train_observations"].update(selected_GT_cycles=522),
    lambda a,b,c:a["interpretation"].update(
        half_open_frame_endpoint_semantics="author_verified"),
    lambda a,b,c:b.update(source_split="test"),
    lambda a,b,c:b["known_protocol"].update(primary_diagnostic_cadence="dense_100ms"),
    lambda a,b,c:b["known_protocol"].update(bootstrap_seed=7),
    lambda a,b,c:b["source_code_sha256"].pop(m48.SOURCES[0]),
    lambda a,b,c:c.update(original_m48_lock_sha256="e"*64),
    lambda a,b,c:c.update(official_test_executed_or_opened=True),
    lambda a,b,c:c.update(research_source_provenance_check_passed=False),
    lambda a,b,c:c["preregistered_future_test_protocol_without_execution"].update(
        test_execution_authorized_by_this_tool=True),
    lambda a,b,c:c["preregistered_future_test_protocol_without_execution"].update(
        future_test_runs_planned=2),
])
def test_m56_refuses_changed_source_or_claim(mutator):
    a,b,c,s,t=fixtures()
    mutator(a,b,c)
    with pytest.raises(ValueError):
        m56.preregister(a,b,c,m55_sha=s,m48_sha=t)

def test_m56_input_path_safety_and_overwrite_refusal(tmp_path,monkeypatch):
    a,b,c,_,_=fixtures()
    m55_path=tmp_path/"m55.json"
    m48_path=tmp_path/"m48.json"
    m55_path.write_text(json.dumps(a),encoding="utf8")
    m48_path.write_text(json.dumps(b),encoding="utf8")
    lock_sha=hashlib.sha256(m48_path.read_bytes()).hexdigest()
    c["original_m48_lock_sha256"]=lock_sha
    called=[]
    def fake_m49_run(**kwargs):
        called.append(kwargs)
        return c
    monkeypatch.setattr(m56.m49,"run",fake_m49_run)
    out=tmp_path/"m56.json"
    report=m56.run(m55_path=m55_path,m48_path=m48_path,
                   m46_path=tmp_path/"not-opened-m46.json",
                   m47_path=tmp_path/"not-opened-m47.json",report_path=out)
    assert out.is_file()
    assert report["test_execution_authorized"] is False
    assert len(called)==1
    assert called[0]["check_only"] is True
    assert out.read_text().startswith('{\n  "protocol": "m56-')
    assert "not-opened-m46" not in out.read_text()
    with pytest.raises(FileExistsError):
        m56.run(m55_path=m55_path,m48_path=m48_path,
                m46_path=tmp_path/"m46",m47_path=tmp_path/"m47",report_path=out)
    link=tmp_path/"alias.json"
    link.symlink_to(m55_path)
    with pytest.raises(ValueError):
        m56._read(link)
    wrong=tmp_path/"wrong.json"
    wrong.write_text("[]",encoding="utf8")
    with pytest.raises(ValueError):
        m56._read(wrong)
