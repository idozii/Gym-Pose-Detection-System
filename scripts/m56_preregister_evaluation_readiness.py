"""M56: strict TRAIN-only pre-registration readiness snapshot. NEVER opens TEST.

Inputs: existing owner M55 aggregate + M48 lock + M49 check-only against
original M46/M47 source/revision. No TAR, CSV, MP4, landmarks, TEST or VALID.
The output is explicitly BLOCKED, not an authorization or test executor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from scripts import m48_freeze_repcount_evidence as m48
from scripts import m49_pretest_git_provenance as m49
from scripts import m55_author_annotation_evidence_review as m55

PROTOCOL = "m56-repcount-preregistered-evaluation-readiness-v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_BYTES = 4 * 1024 * 1024
DEFAULT_M55 = Path("data/processed/m55_author_evidence_review.json")
DEFAULT_M48 = Path("data/processed/m48_train_evidence_freeze.json")
DEFAULT_REPORT = Path("data/processed/m56_test_readiness_blocked.json")

BLOCKERS = (
    "authoritative_original_archive_release_and_digest",
    "written_data_and_video_use_permissions",
    "human_annotation_boundary_and_endpoint_policy",
    "official_test_source_manifest_precommitted_without_peeking",
    "authoritative_subject_session_camera_overlap_policy",
    "GT_to_decoded_video_frame_timing_check_before_outcome_review",
    "independent_zero_rep_negative_control_plan",
)

def _read(path: Path) -> tuple[dict, str]:
    if (not path.is_file() or path.is_symlink() or
            path.stat().st_size > MAX_BYTES):
        raise ValueError("M56 input must be a small regular owner-local JSON")
    raw=path.read_bytes()
    value=json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("M56 requires JSON object reports")
    return value, hashlib.sha256(raw).hexdigest()

def preregister(m55_report: dict, m48_lock: dict,
                m49_check: dict, *, m55_sha: str, m48_sha: str) -> dict:
    """Fail closed on provenance, source/replay parity and TRAIN scope."""
    if not all(isinstance(x,dict) for x in (m55_report,m48_lock,m49_check)):
        raise ValueError("M56 requires three source evidence objects")
    if (not all(isinstance(x,str) and SHA256.fullmatch(x)
                for x in (m55_sha,m48_sha)) or
        m55_report.get("protocol") != m55.PROTOCOL or
        m55_report.get("source_split") != "train" or
        m55_report.get("m54_sha256_matches_original_m48_lock") is not True or
        m55_report.get("source_videos_filenames_gt_boundaries_persisted") is not False or
        m55_report.get("official_test_or_validation_opened") is not False or
        m55_report.get("author_contact_or_permission_claimed") is not False or
        m48_lock.get("protocol") != m48.PROTOCOL or
        m48_lock.get("source_split") != "train" or
        m48_lock.get("owner_observed_evidence_locked") is not True or
        m49_check.get("protocol") != m49.PROTOCOL or
        m49_check.get("source_split_used_by_m49") != "train" or
        m49_check.get("original_m48_lock_sha256") != m48_sha or
        m49_check.get("research_source_provenance_check_passed") is not True or
        m49_check.get("external_test_readiness_approved") is not False or
        m49_check.get("official_test_executed_or_opened") is not False or
        m49_check.get("validation_or_test_media_scanned") is not False or
        m49_check.get("no_original_counter_modified") is not True):
        raise ValueError("M56 locked TRAIN provenance or TEST embargo mismatch")

    train_sha=m48_lock.get("publisher_train_annotation_sha256")
    if not isinstance(train_sha,str) or not SHA256.fullmatch(train_sha) or (
        m49_check.get("frozen_training_annotation_sha256") != train_sha):
        raise ValueError("M56 original TRAIN annotation checksum mismatch")
    fingerprints=m48_lock.get("source_code_sha256")
    if (not isinstance(fingerprints,dict) or
        set(fingerprints) != set(m48.SOURCES) or
        any(not isinstance(v,str) or not SHA256.fullmatch(v)
            for v in fingerprints.values())):
        raise ValueError("M56 requires 17 frozen source fingerprints")

    proto=m48_lock.get("known_protocol")
    intent=m49_check.get("preregistered_future_test_protocol_without_execution")
    claims=m55_report.get("claim_gate")
    evidence=m55_report.get("owner_original_train_observations")
    semantics=m55_report.get("interpretation")
    if any(not isinstance(x,dict) for x in (proto,intent,claims,evidence,semantics)):
        raise ValueError("M56 requires original frozen study/claim protocol")
    if (proto.get("selected_method") != "relative_excursion" or
        proto.get("control_method") != "low_to_high" or
        proto.get("source_split") != "train" or
        proto.get("primary_diagnostic_cadence") != "dense_200ms" or
        proto.get("secondary_cadence") != "subsample_400ms" or
        proto.get("original_gt_event_matcher") != "m37_one_to_one_gt_end_tolerance" or
        proto.get("cohort_video_count") != 20 or
        proto.get("ground_truth_reps") != 209 or
        proto.get("bootstrap_unit") != "source_video" or
        proto.get("bootstrap_draws") != 5000 or
        proto.get("bootstrap_seed") != 2026 or
        proto.get("single_predeclared_test_execution_permitted_by_this_script") is not False or
        intent.get("test_execution_authorized_by_this_tool") is not False or
        intent.get("future_no_test_adaptation") is not True or
        intent.get("future_test_runs_planned") != 1 or
        evidence.get("squat_source_videos") != 101 or
        evidence.get("zero_GT_squat_videos") != 0 or
        evidence.get("selected_source_videos") != 40 or
        evidence.get("selected_GT_cycles") != 521 or
        evidence.get("zero_count_false_positive_specificity_unmeasured") is not True or
        semantics.get("half_open_frame_endpoint_semantics") !=
            "project_parser_convention_not_author_verified" or
        semantics.get("same_subject_or_session_across_filenames") != "unverified"):
        raise ValueError("M56 frozen TRAIN evaluation design mismatch")
    required_claims = (
        "can_claim_official_original_archive_verified",
        "can_claim_commercial_or_video_redistribution_license",
        "can_claim_subject_disjoint_generalization",
        "can_claim_gt_boundaries_match_anatomical_phases",
        "can_claim_zero_rep_specificity",
        "can_claim_publisher_test_performance",
    )
    if (any(claims.get(k) is not False for k in required_claims) or
        claims.get("can_report_m54_train_aggregate_consistency") is not True):
        raise ValueError("M56 refuses unsupported scientific claim upgrades")

    return {
        "protocol":PROTOCOL,
        "status":"BLOCKED_REQUIRES_EXTERNAL_AUTHORIZATION_AND_PREREGISTRATION",
        "input_split":"train",
        "m55_owner_report_sha256":m55_sha,
        "m48_owner_lock_sha256":m48_sha,
        "frozen_sources_confirmed_by_original_M49_check_only":True,
        "publisher_test_accessed":False,
        "publisher_validation_accessed":False,
        "test_execution_authorized":False,
        "test_manifest_inspected_or_saved":False,
        "future_study_preregistration_proposal":{
            "cohort":"all eligible OFFICIAL Squat TEST entries; do not preselect by outcomes",
            "test_cohort_manifest_status":"NOT_DEFINED_OR_ACCESSED",
            "candidate":"frozen M45 relative_excursion",
            "control":"frozen M39 low_to_high",
            "primary_condition":"dense_200ms, single offline Python MediaPipe pose history",
            "secondary_condition":"400ms every-other observation from SAME dense pose history",
            "primary_metric":"micro event F1, one-to-one original M37 publisher-GT cycle-end matching",
            "original_m37_matching_window":
                "max(750ms, 35% annotated cycle duration), capped at 2500ms",
            "event_matching_policy":
                "one GT-to-prediction assignment at most once; unmatched predictions FP and GT FN",
            "primary_metric_undefined_policy":
                "report undefined if global F1 denominator is zero; never silently set to 1",
            "required_secondary_metrics":[
                "micro event TP,FP,FN,precision,recall",
                "absolute per-video count MAE",
                "OBO |predicted count-GT count|<=1 fraction",
                "exact-count fraction",
                "unmatched FP events per source video",
                "GT total, evaluated/excluded counts and reason totals"
            ],
            "positive_gt_only_secondary_metric":
                "mean per-video absolute count difference divided by positive GT; undefined for GT=0",
            "negative_gt_policy":
                "if authorized TEST contains GT=0 Squat, include in count MAE/OBO/exact and FP; exclude only from positive-GT normalized MAE with separate denominator; no imputation",
            "uncertainty":
                "predeclared 5000-draw paired percentile 95% bootstrap, seed 2026; source-video clusters only if subject grouping unavailable, with dependency caveat",
            "exclusion_and_missing_video_policy":
                "inventory and report ALL eligible failures/codec/label mismatches, no outcome-based exclusions; resolve before any authorized one-shot execution",
            "stopping_rule":"one authorized full-cohort pass only; no TEST-tuned thresholds, model selection or reruns to improve scores",
            "exploratory_analyses":
                "200ms vs 400ms and subgroup breakdowns explicitly secondary; no post hoc confirmatory claims",
            "known_limits":[
                "TRAIN file hashes do not guarantee subject/session independence",
                "M38 previously inspected VALID clips are not untouched held-out evidence",
                "publisher boundaries not independently anatomical phase labels",
                "browser MediaPipe and Python Solutions pose pipelines not validated as equivalent",
                "TRAIN Squat provides zero count-zero examples; specificity cannot be inferred"
            ]
        },
        "manual_external_blockers":{name:"unverified" for name in BLOCKERS},
        "claims_permitted_now":[
            "reproducible TRAIN-only source/protocol evidence",
            "author-described general repetition-cycle annotation protocol"
        ],
        "claims_blocked":[
            "official TEST performance", "new externally validated accuracy",
            "subject-disjoint generalization", "negative-only specificity",
            "phase-specific GT correctness", "commercial/redistribution rights",
            "medical or fitness coaching safety"
        ],
        "data_media_decoded_or_extracted":False,
        "source_file_names_or_gt_intervals_persisted":False
    }

def run(*,m55_path:Path,m48_path:Path,m46_path:Path,m47_path:Path,
        report_path:Path) -> dict:
    if report_path.exists():
        raise FileExistsError("M56 refuses overwrite of existing preregistration")
    m55_report,m55_sha=_read(m55_path)
    m48_lock,m48_sha=_read(m48_path)
    # Delegated to ORIGINAL M49 read-only Git and M48 source/replay-byte gates.
    # This never opens any publisher TEST or VALID material.
    proof=m49.run(m46_path=m46_path,m47_path=m47_path,
                  lock_path=m48_path,report_path=report_path,check_only=True)
    result=preregister(m55_report,m48_lock,proof,m55_sha=m55_sha,m48_sha=m48_sha)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open("x",encoding="utf8") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")
    return result

def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m55-report",type=Path,default=DEFAULT_M55)
    p.add_argument("--m48-lock",type=Path,default=DEFAULT_M48)
    p.add_argument("--m46-report",type=Path,default=m48.DEFAULT_M46)
    p.add_argument("--m47-report",type=Path,default=m48.DEFAULT_M47)
    p.add_argument("--report",type=Path,default=DEFAULT_REPORT)
    a=p.parse_args(argv)
    try:
        result=run(m55_path=a.m55_report,m48_path=a.m48_lock,
                   m46_path=a.m46_report,m47_path=a.m47_report,report_path=a.report)
    except (OSError,ValueError,TypeError,KeyError,UnicodeError) as exc:
        print("M56 BLOCKED / refused: "+str(exc),file=sys.stderr)
        return 1
    print(json.dumps({
        "protocol":result["protocol"],
        "status":result["status"],
        "train_only":True,
        "frozen_source_provenance_check_passed":True,
        "test_execution_authorized":False,
        "publisher_test_accessed":False,
        "manual_blockers":list(result["manual_external_blockers"]),
        "report_path":str(a.report)
    },indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
