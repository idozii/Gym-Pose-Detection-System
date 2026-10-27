"""M29 source-cohort gate tests: synthetic private files, no actual people."""
from __future__ import annotations
import csv
import json

import pytest

from scripts.m29_external_pushup_cohort import (
    COLUMNS, SCHEMA_VERSION, audit_cohort, init_manifest, main
)


def setup_samples(tmp_path):
    root=tmp_path/"private_videos"
    root.mkdir(exist_ok=True)
    rows=[]
    for person in range(5):
        for view in ("front","side","diagonal"):
            for label in ("good","bad"):
                clip=f"extclip_{person:08d}_{view}_{label}"
                name=clip+".mp4"
                # Byte-unique synthetic files only, not real media.
                (root/name).write_bytes(("synthetic "+clip).encode())
                rows.append({
                    "video_id":clip,"subject_id":f"extperson_{person:08d}",
                    "view":view,"source_file":name,
                    "reviewer_a_id":"reviewer_A","reviewer_a_label":label,
                    "reviewer_b_id":"reviewer_B","reviewer_b_label":label,
                    "final_label":label,"adjudicator_id":"",
                    "consent_ref":f"consent_{person:08d}",
                    "age_18plus_verified":"yes",
                    "annotation_version":SCHEMA_VERSION
                })
    manifest=tmp_path/"private_manifest.csv"
    with manifest.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return manifest,root,rows


def rewrite_manifest(manifest,rows):
    with manifest.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def test_m29_private_external_cohort_summary_only(tmp_path):
    manifest,root,rows=setup_samples(tmp_path)
    result=audit_cohort(manifest,root)
    assert result["annotated_clips"]==30
    assert result["distinct_pseudonymous_participants"]==5
    assert result["two_reviewer_agreements"]==30
    assert result["third_reviewer_adjudications"]==0
    assert result["ready_for_pilot"] is True
    assert result["ready_for_large_independent_study"] is False
    assert result["external_source_subject_independence_verified"] is False
    assert result["model_predictions_or_labels_from_original_TEST_accessed"] is False
    assert all(result["by_view_and_label"].values())
    serialized=json.dumps(result)
    assert "extperson_" not in serialized
    assert "extclip_" not in serialized
    assert "consent_00000000" not in serialized
    assert "private_videos" not in serialized


def test_m29_detects_double_content_and_repeated_clip(tmp_path):
    manifest,root,rows=setup_samples(tmp_path)
    rows[-1]["source_file"]=rows[0]["source_file"]
    rewrite_manifest(manifest,rows)
    with pytest.raises(ValueError,match="Duplicate source_file"):
        audit_cohort(manifest,root)
    manifest,root,rows=setup_samples(tmp_path)
    (root/rows[-1]["source_file"]).write_bytes(
        (root/rows[0]["source_file"]).read_bytes()
    )
    with pytest.raises(ValueError,match="Duplicate video bytes"):
        audit_cohort(manifest,root)


def test_m29_requires_independent_label_review_and_consensus(tmp_path):
    manifest,root,rows=setup_samples(tmp_path)
    rows[0]["reviewer_b_label"]="bad"
    with pytest.raises(ValueError,match="third independent adjudicator"):
        rewrite_manifest(manifest,rows)
        audit_cohort(manifest,root)
    rows[0]["adjudicator_id"]="reviewer_C"
    rewrite_manifest(manifest,rows)
    assert audit_cohort(manifest,root)["third_reviewer_adjudications"]==1
    rows[0]["reviewer_a_id"]="reviewer_B"
    rewrite_manifest(manifest,rows)
    with pytest.raises(ValueError,match="Two distinct"):
        audit_cohort(manifest,root)


def test_m29_fails_closed_on_consent_and_age(tmp_path):
    manifest,root,rows=setup_samples(tmp_path)
    rows[0]["age_18plus_verified"]="no"
    rewrite_manifest(manifest,rows)
    with pytest.raises(ValueError,match="adult volunteer"):
        audit_cohort(manifest,root)
    rows[0]["age_18plus_verified"]="yes"
    rows[0]["consent_ref"]=""
    rewrite_manifest(manifest,rows)
    with pytest.raises(ValueError,match="consent reference"):
        audit_cohort(manifest,root)


def test_m29_rejects_absolute_and_outside_paths(tmp_path):
    manifest,root,rows=setup_samples(tmp_path)
    rows[0]["source_file"]="../outside.mp4"
    rewrite_manifest(manifest,rows)
    with pytest.raises(ValueError,match="safe relative"):
        audit_cohort(manifest,root)
    rows[0]["source_file"]="/tmp/secret.mp4"
    rewrite_manifest(manifest,rows)
    with pytest.raises(ValueError,match="safe relative"):
        audit_cohort(manifest,root)
    rows[0]["source_file"]="missing.mp4"
    rewrite_manifest(manifest,rows)
    with pytest.raises(ValueError,match="Video absent"):
        audit_cohort(manifest,root)


def test_m29_init_local_manifest_no_overwrite_and_strict_version(tmp_path):
    empty=tmp_path/"external"/"manifest.csv"
    created=init_manifest(empty)
    assert created["rows"]==0
    assert tuple(next(csv.reader(empty.open())))==COLUMNS
    with pytest.raises(FileExistsError):
        init_manifest(empty)
    manifest,root,rows=setup_samples(tmp_path)
    rows[0]["annotation_version"]="unknown"
    rewrite_manifest(manifest,rows)
    with pytest.raises(ValueError,match="annotation rubric version"):
        audit_cohort(manifest,root)


def test_m29_cli_audit_no_raw_files_in_report(tmp_path):
    manifest,root,rows=setup_samples(tmp_path)
    report=tmp_path/"private_report.json"
    assert main(["audit","--manifest",str(manifest),
                 "--root",str(root),"--report",str(report)])==0
    assert report.exists()
    stored=report.read_text()
    assert "extclip_00000000" not in stored
    assert "consent_" not in stored
    assert main(["audit","--manifest",str(manifest),
                 "--root",str(root),"--report",str(report)])==1
