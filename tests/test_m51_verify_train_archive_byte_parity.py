"""M51 synthetic publisher TAR video-content comparison without real media."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile

import pytest

from scripts import m51_verify_train_archive_byte_parity as m51
from scripts.m50_train_label_video_integrity import _publisher_cohorts


def _add(tf, name, data):
    entry = tarfile.TarInfo(name)
    entry.size = len(data)
    tf.addfile(entry, io.BytesIO(data))


def _source_csv():
    names = [f"synthetic_squat_{n:03d}.mp4" for n in range(40)]
    salt = b"repcount-m40-train-v1:2026:"
    ranked = sorted(names,key=lambda name:(
        hashlib.sha256(salt+name.encode("ascii")).digest(),name
    ))
    idx = {name:i for i,name in enumerate(ranked)}
    buf = io.StringIO()
    fields = ["name","type","count"] + [f"L{i}" for i in range(1,55)]
    writer=csv.DictWriter(buf,fieldnames=fields)
    writer.writeheader()
    for name in reversed(names):
        rank=idx[name]
        reps = (27 if rank == 0 else 15) if rank < 20 else (
            19 if rank == 20 else 10
        )
        row={"name":name,"type":"Squat","count":reps}
        for i in range(reps):
            row[f"L{2*i+1}"]=i*8
            row[f"L{2*i+2}"]=i*8+4
        writer.writerow(row)
    return buf.getvalue().encode("utf8")


def _fixture(tmp_path, *, wrong_copy=None, omit=None):
    raw=_source_csv()
    cohorts=_publisher_cohorts(raw)
    archive=tmp_path/"RepCountA.tar.gz"
    dev=tmp_path/"m45"
    repl=tmp_path/"m46"
    lookup={}
    for index,rows in enumerate(cohorts):
        destination=(dev if index==0 else repl)/"official_tar_selected"/"train"
        destination.mkdir(parents=True)
        for row in rows:
            name=row["video"]
            content=(b"R"+name.encode("ascii")*1000)[:15000]
            if len(content)<10000:
                content=content+b"z"*(10000-len(content))
            lookup[name]=content
            destination.joinpath(name).write_bytes(
                (content[:-1]+b"Z") if name == wrong_copy else content
            )
    with tarfile.open(archive,"w:gz") as tf:
        # Mimic TAR containing TEST and validation files alongside TRAIN.
        # Selected train clips do not depend on TAR member order.
        for name,content in reversed(list(lookup.items())):
            if name != omit:
                _add(tf,"LLSP/video/train/"+name,content)
            _add(tf,"LLSP/video/valid/"+name,b"v"*16000)
            _add(tf,"LLSP/video/test/"+name,b"t"*16000)
        _add(tf,"LLSP/annotation/test.csv",b"do not parse test")
        _add(tf,"LLSP/annotation/valid.csv",b"do not parse valid")
        _add(tf,"LLSP/annotation/train.csv",raw)
    lock=tmp_path/"M48.json"
    lock.write_text(json.dumps({
        "publisher_train_annotation_sha256":hashlib.sha256(raw).hexdigest()
    }))
    timing=tmp_path/"M50.json"
    timing.write_text(json.dumps({
        "protocol":"m50-squat-train-annotation-video-timing-audit-v1",
        "source_split":"train",
        "publisher_train_annotation_sha256":hashlib.sha256(raw).hexdigest(),
        "original_m48_lock_sha256":hashlib.sha256(lock.read_bytes()).hexdigest(),
        "videos_total":40,"annotated_gt_reps_total":521,
        "critical_timing_or_decode_integrity_flags":0,
        "validation_or_test_scanned":False,
        "detector_or_matcher_modified":False,
        "cohort_video_timing":[
            {"cohort":"original_development_20","videos":20,"annotated_reps":312},
            {"cohort":"frozen_replication_20","videos":20,"annotated_reps":209}
        ]
    }))
    return archive,dev,repl,lock,timing,cohorts


def _params(tmp_path, fixture):
    archive,dev,repl,lock,timing,cohorts=fixture
    return {
        "archive":archive,"development_root":dev,
        "replication_root":repl,
        "m46_path":tmp_path/"unused_m46.json",
        "m47_path":tmp_path/"unused_m47.json",
        "m48_lock":lock,
        "m50_path":timing,
        "report":tmp_path/"M51.json",
        "freeze_gate":lambda **kwargs:{
            "research_source_provenance_check_passed":True
        }
    }


def test_m51_train_only_40_source_copies_identical_with_late_train_csv(tmp_path):
    fixture=_fixture(tmp_path)
    kwargs=_params(tmp_path,fixture)
    result=m51.run(**kwargs)
    assert result["videos_byte_identical_total"]==40
    assert result["annotated_gt_reps_total"]==521
    assert result["validation_or_test_video_payload_hashed_or_decoded"] is False
    assert result["validation_or_test_annotation_csv_loaded"] is False
    assert result["tar_archive_headers_across_splits_traversed"] is True
    assert result["research_model_modified"] is False
    assert [(x["videos_byte_identical_to_source_tar"],x["annotated_gt_reps"])
            for x in result["cohorts"]]==[(20,312),(20,209)]
    text=kwargs["report"].read_text()
    for private in ("synthetic_squat_",'"video_path"',
                    '"cycle_bounds_frames"','"landmarks"',
                    '"frames"','"time_ms"'):
        assert private not in text
    with pytest.raises(FileExistsError,match="refusing overwrite"):
        m51.run(**kwargs)


def test_m51_rejects_same_length_bitwise_modified_local_video(tmp_path):
    raw=_source_csv()
    first=_publisher_cohorts(raw)[0][0]["video"]
    fixture=_fixture(tmp_path,wrong_copy=first)
    kwargs=_params(tmp_path,fixture)
    with pytest.raises(ValueError, match="does not match"):
        m51.run(**kwargs)
    assert not kwargs["report"].exists()


def test_m51_rejects_missing_selected_tar_member(tmp_path):
    first=_publisher_cohorts(_source_csv())[1][1]["video"]
    fixture=_fixture(tmp_path,omit=first)
    kwargs=_params(tmp_path,fixture)
    with pytest.raises(ValueError, match="not all forty"):
        m51.run(**kwargs)
    assert not kwargs["report"].exists()


def test_m51_bad_existing_m50_or_m48_hash_causes_no_new_report(tmp_path):
    fixture=_fixture(tmp_path)
    kwargs=_params(tmp_path,fixture)
    orig=json.loads(kwargs["m50_path"].read_text())
    changed={**orig,"critical_timing_or_decode_integrity_flags":1}
    kwargs["m50_path"].write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="M50 TRAIN report"):
        m51.run(**kwargs)
    kwargs["m50_path"].write_text(json.dumps(orig))
    kwargs["m48_lock"].write_text(json.dumps({
        "publisher_train_annotation_sha256":"f"*64
    }))
    with pytest.raises(ValueError,match="CSV SHA256 differs"):
        m51.run(**kwargs)
    assert not kwargs["report"].exists()


def test_m51_raw_stream_requires_full_declared_byte_length():
    with pytest.raises(ValueError,match="truncated"):
        m51._hash_stream(io.BytesIO(b"x"*10000),expected_size=11000)
    with pytest.raises(ValueError,match="size is invalid"):
        m51._hash_stream(io.BytesIO(b"x"),expected_size=1)
    with pytest.raises(ValueError,match="exceeds declared"):
        m51._hash_stream(io.BytesIO(b"x"*10011),expected_size=10010)


def test_m51_verification_gate_precedes_archive_read(tmp_path):
    fixture=_fixture(tmp_path)
    kwargs=_params(tmp_path,fixture)
    touched=[False]
    def forbid_csv(*args):
        touched[0]=True
        raise AssertionError("M51 read archive before failed pretest gate")
    kwargs["source_csv_fn"]=forbid_csv
    kwargs["freeze_gate"]=lambda **kw:{
        "research_source_provenance_check_passed":False
    }
    with pytest.raises(ValueError, match="verified original M49"):
        m51.run(**kwargs)
    assert touched==[False]
    assert not kwargs["report"].exists()
