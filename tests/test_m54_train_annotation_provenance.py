"""M54 synthetic TRAIN-only annotation semantics tests, no publisher media."""
import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts import m54_train_annotation_provenance as m54


def train_csv(*,include_zero=False):
    """Construct 40 original hash-cohort records plus one unselected zero."""
    names=[f"synthetic{i:03d}.mp4" for i in range(41 if include_zero else 40)]
    salt=b"repcount-m40-train-v1:2026:"
    ranked=sorted(names,key=lambda name:(hashlib.sha256(
        salt+name.encode("ascii")).digest(),name))
    selected=ranked[:40]
    records={}
    for i,name in enumerate(selected):
        gt=16 if i<12 else 15 if i<20 else 11 if i<29 else 10
        # original 20 GT=12*16+8*15=312; second=9*11+11*10=209
        records[name]=gt
    if include_zero:
        records[ranked[-1]]=0
    buf=io.StringIO()
    fields=["name","type","count"]+[f"L{i}" for i in range(1,33)]
    writer=csv.DictWriter(buf,fieldnames=fields)
    writer.writeheader()
    for name,count in records.items():
        data={"name":name,"type":"squat","count":str(count)}
        for idx in range(count):
            data[f"L{2*idx+1}"]=str(2+20*idx)
            data[f"L{2*idx+2}"]=str(15+20*idx)
        writer.writerow(data)
    return buf.getvalue().encode("utf8")


def test_m54_40_original_squat_train_hash_cohorts_and_schema():
    result=m54.audit_train_csv(train_csv())
    groups=result["original_frozen_cohorts"]
    assert set(groups)=={"development_20","replication_20"}
    assert result["train_squat_annotation_rows"]==40
    assert result["train_squat_zero_gt_videos"]==0
    assert result["frame_bounds_are_ordered_nonoverlapping_for_squat_train"]
    assert result["gt_frame_position_is_not_an_anatomic_phase_label"] is True
    assert result["original_half_open_interpretation_verified_against_human_annotator_policy"] is False
    assert groups["development_20"]["annotated_repetitions"]==312
    assert groups["replication_20"]["annotated_repetitions"]==209
    assert groups["development_20"]["total_valid_positive_frame_spans"]==312
    assert groups["replication_20"]["total_valid_positive_frame_spans"]==209
    assert groups["development_20"]["smallest_cycle_span_frames"]==13
    assert groups["replication_20"]["largest_cycle_span_frames"]==13
    assert groups["development_20"]["adjacent_cycle_gaps_with_positive_unlabeled_frames"]==292
    assert not result["recognized_explicit_person_session_camera_schema"]["subject"]["recognized_explicit_column"]
    assert "L1" in result["publisher_train_csv_column_names"]
    assert not any(x in result for x in ["video","filename","person_id","rows"])


def test_m54_zero_rep_train_controls_can_exist_outside_selected_40_without_false_assumption():
    result=m54.audit_train_csv(train_csv(include_zero=True))
    assert result["train_squat_annotation_rows"]==41
    assert result["train_squat_zero_gt_videos"]==1
    assert result["train_squat_positive_gt_videos"]==40
    for group in result["original_frozen_cohorts"].values():
        assert group["source_files"]==20
        assert group["zero_gt_videos"]==0


def test_m54_original_train_sha_and_M49_gate_no_test_or_reproducibility_relabel(tmp_path):
    raw=train_csv(include_zero=True)
    sha=hashlib.sha256(raw).hexdigest()
    lock=tmp_path/"m48.json"
    lock.write_text(json.dumps({
        "source_split":"train","publisher_train_annotation_sha256":sha
    }),encoding="utf8")
    out=tmp_path/"m54.json"
    ref=tmp_path/"m46.json"
    r47=tmp_path/"m47.json"
    ref.write_text("{}",encoding="utf8")
    r47.write_text("{}",encoding="utf8")
    def verify(**kwargs):
        assert kwargs["check_only"] is True
        return {
            "research_source_provenance_check_passed":True,
            "official_test_executed_or_opened":False
        }
    actual=m54.run(
        archive=tmp_path/"source.tar.gz",report=out,
        m46=ref,m47=r47,lock=lock,
        csv_reader=lambda archive,split:(
            raw if split=="train" else (_ for _ in ()).throw(
                AssertionError("forbidden split"))),
        verifier=verify
    )
    assert actual["source_split"]=="train"
    assert actual["official_publisher_archive_byte_parity_verified"] is False
    assert actual["video_bytes_extracted_or_decoded"] is False
    assert actual["test_or_validation_csv_or_media_opened"] is False
    assert actual["filename_ids_or_cycle_boundaries_saved"] is False
    assert actual["license_and_provenance_blockers"]["commercial_use_permission"]=="unverified"
    assert actual["original_train_annotation_sha256"]==sha
    assert "synthetic" not in out.read_text()
    assert len(json.loads(out.read_text())["train_label_aggregate"][
        "original_frozen_cohorts"])==2
    with pytest.raises(FileExistsError):
        m54.run(archive=tmp_path/"source.tar.gz",report=out,
                m46=ref,m47=r47,lock=lock,verifier=verify,
                csv_reader=lambda *_:raw)


def test_m54_bad_original_csv_sha_rejected_without_output(tmp_path):
    raw=train_csv()
    lock=tmp_path/"m48.json"
    lock.write_text(json.dumps({
        "source_split":"train","publisher_train_annotation_sha256":"a"*64
    }),encoding="utf8")
    report=tmp_path/"m54.json"
    with pytest.raises(ValueError,match="CSV SHA"):
        m54.run(archive=tmp_path/"original.tar.gz",report=report,
                m46=tmp_path/"m46",m47=tmp_path/"m47",lock=lock,
                csv_reader=lambda *_:raw,verifier=lambda **kwargs:{
                    "research_source_provenance_check_passed":True,
                    "official_test_executed_or_opened":False
                })
    assert not report.exists()


def test_m54_does_not_touch_media_or_claim_license_by_assumption():
    source=Path(m54.__file__).read_text(encoding="utf8")
    assert "parse_official_csv" in source
    assert '_embedded_csv' in source
    assert 'm48.get("publisher_train_annotation_sha256")' in source
    assert 'csv_reader or _embedded_csv' in source
    assert 'video_bytes_extracted_or_decoded":False' in source
    assert 'source_split":"train"' in source
    assert "cv2.VideoCapture" not in source
    assert "extractfile" not in source
    assert "test_or_validation_csv_or_media_opened" in source
    assert set(m54.PUBLIC_AUTHOR_SOURCES)=={"official_dataset","official_repo"}
