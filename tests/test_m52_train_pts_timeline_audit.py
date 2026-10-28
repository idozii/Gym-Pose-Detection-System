"""M52 synthetic TRAIN-only PTS tests; no publisher data or real video."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

import pytest

from scripts import m52_train_pts_timeline_audit as m52
from scripts.m50_train_label_video_integrity import _publisher_cohorts


def _training_csv():
    names=[f"squat_pts_{i:03d}.mp4" for i in range(40)]
    salt=b"repcount-m40-train-v1:2026:"
    ranked=sorted(names,key=lambda name:(
        hashlib.sha256(salt+name.encode()).digest(),name
    ))
    rank={name:i for i,name in enumerate(ranked)}
    cols=["name","type","count"]+[f"L{k}" for k in range(1,55)]
    output=io.StringIO()
    writer=csv.DictWriter(output,fieldnames=cols)
    writer.writeheader()
    for name in reversed(names):
        i=rank[name]
        count=(27 if i==0 else 15) if i<20 else (
            19 if i==20 else 10
        )
        record={"name":name,"type":"squat","count":count}
        for j in range(count):
            record[f"L{2*j+1}"]=j*6
            record[f"L{2*j+2}"]=j*6+4
        writer.writerow(record)
    return output.getvalue().encode()


def _prepare(tmp_path):
    raw=_training_csv()
    groups=_publisher_cohorts(raw)
    dev=tmp_path/"m45"
    repl=tmp_path/"m46"
    for root,cohort in zip((dev,repl),groups):
        sub=root/"official_tar_selected"/"train"
        sub.mkdir(parents=True)
        for item in cohort:
            (sub/item["video"]).write_bytes(b"existing-train-video")
    archive=tmp_path/"RepCountA.tar.gz"
    archive.write_bytes(b"synthetic-source-reference")
    lock=tmp_path/"m48.json"
    lock.write_text(json.dumps({
        "publisher_train_annotation_sha256":hashlib.sha256(raw).hexdigest()
    }))
    m50=tmp_path/"m50.json"
    m50.write_text(json.dumps({
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
    m51=tmp_path/"m51.json"
    m51.write_text(json.dumps({
        "protocol": "m51-train-selected-archive-byte-parity-v1",
        "publisher_train_annotation_sha256":hashlib.sha256(raw).hexdigest(),
        "original_m48_lock_sha256":hashlib.sha256(lock.read_bytes()).hexdigest(),
        "original_m50_report_sha256":hashlib.sha256(m50.read_bytes()).hexdigest(),
        "source_split_video_contents_hashed":"train_only",
        "videos_byte_identical_total":40,
        "annotated_gt_reps_total":521,
        "verification_status":"all_40_train_local_mp4_bytes_equal_tar_members",
        "test_evaluation_performed":False,
        "validation_or_test_video_payload_hashed_or_decoded":False,
        "cohorts":[
            {"cohort":"original_development_20",
             "videos_byte_identical_to_source_tar":20,
             "annotated_gt_reps":312},
            {"cohort":"frozen_replication_20",
             "videos_byte_identical_to_source_tar":20,
             "annotated_gt_reps":209}
        ]
    }))
    return raw,dict(
        archive=archive,
        development_root=dev,replication_root=repl,
        m46_path=tmp_path/"m46.json",m47_path=tmp_path/"m47.json",
        m48_lock=lock,m50_path=m50,m51_path=m51,
        report=tmp_path/"m52.json",
        csv_reader=lambda archive,split:raw if split=="train" else None,
        pts_probe=lambda video:[k/25 for k in range(1000)],
        fps_probe=lambda video:(25.,1000),
        evidence_gate=lambda **k:{"research_source_provenance_check_passed":True}
    )


def test_m52_exact_constant_fps_has_zero_pts_timing_drift():
    seq=[k/25 for k in range(20)]
    report=m52.audit_timeline(
        gt_bounds=[1,5,6,19],fps=25.,declared_frames=20,
        pts=seq
    )
    assert report["GT_windows"]==2
    assert report["GT_boundaries_with_abs_drift_gt50ms"]==0
    assert report["irregular_interframe_steps_gt20pct_from_median"]==0
    assert report["max_abs_timeline_drift_ms"]<1e-9
    assert report["exclusive_GT_ends_without_corresponding_PTS_frame"]==0
    assert report["median_derived_fps_deviation_pct"]<1e-8


def test_m52_detects_variable_pts_and_never_extrapolates_missing_exclusive_end():
    pts=[0.,.04,.08,.12,.16,.20,.24,.35,.39,.43]
    data=m52.audit_timeline(
        gt_bounds=[0,5,7,10],fps=25.,declared_frames=10,pts=pts
    )
    assert data["GT_windows"]==2
    assert data["exclusive_GT_ends_without_corresponding_PTS_frame"]==1
    assert data["observable_GT_boundary_points"]==3
    assert data["GT_boundaries_with_abs_drift_gt50ms"]==1
    assert data["max_abs_timeline_drift_ms"]==pytest.approx(70.)
    assert data["irregular_interframe_steps_gt20pct_from_median"]==1
    with pytest.raises(ValueError,match="frame count differs"):
        m52.audit_timeline(
            gt_bounds=[1,5],fps=25.,declared_frames=11,pts=pts
        )
    with pytest.raises(ValueError,match="strictly increasing"):
        m52.audit_timeline(
            gt_bounds=[1,5],fps=25.,declared_frames=10,
            pts=[0]*10
        )


class Result:
    def __init__(self, data, rc=0):
        self.stdout=json.dumps(data)
        self.stderr=""
        self.returncode=rc


def test_m52_ffprobe_uses_video_only_show_frames_and_rejects_missing_pts(tmp_path):
    video=tmp_path/"clip.mp4"
    video.write_bytes(b"x"*11000)
    commands=[]
    def mock_run(cmd, **kwargs):
        commands.append((cmd,kwargs))
        return Result({"frames":[{
            "best_effort_timestamp_time":f"{i/25:.6f}"
        } for i in range(10)]})
    out=m52.probe_pts(video,runner=mock_run)
    assert len(out)==10
    assert commands[0][0][:6]==[
        "ffprobe","-v","error","-select_streams","v:0","-show_frames"
    ]
    assert "frame=best_effort_timestamp_time" in commands[0][0]
    assert commands[0][0][-2:]==["-i",str(video)]
    def no_pts(cmd,**kwargs):
        return Result({"frames":[{
            "best_effort_timestamp_time":("N/A" if i==2 else str(i*.04))
        } for i in range(10)]})
    with pytest.raises(ValueError,match="nonnumeric"):
        m52.probe_pts(video,runner=no_pts)
    def wrong_order(cmd,**kwargs):
        return Result({"frames":[{
            "best_effort_timestamp_time":str(.04*(i if i<8 else 7))
        } for i in range(10)]})
    with pytest.raises(ValueError,match="not strictly increasing"):
        m52.probe_pts(video,runner=wrong_order)


def test_m52_pipeline_checks_original_m50_m51_and_persists_only_aggregates(tmp_path):
    raw,args=_prepare(tmp_path)
    result=m52.run(**args)
    assert result["videos_evaluated"]==40
    assert result["annotated_gt_events"]==521
    assert result["PTS_timeline_anomalies_descriptive"]==0
    assert result["test_or_validation_media_scanned"] is False
    assert result["deployed_counter_modified"] is False
    assert result["per_frame_data_persisted"] is False
    assert [x["GT_reps"] for x in result["cohort_video_timeline"]]==[312,209]
    assert all(x["GT_boundary_points_with_abs_drift_gt50ms"] == 0
               for x in result["cohort_video_timeline"])
    report_text=args["report"].read_text()
    for forbidden in ("squat_pts_000.mp4",'"frames"',
                      '"cycle_bounds_frames"', '"time_ms"',
                      '"landmarks"', '"video_path"'):
        assert forbidden not in report_text
    with pytest.raises(FileExistsError, match="exists"):
        m52.run(**args)


def test_m52_rejects_corrupt_prior_m51_and_bad_source_sha_before_pts(tmp_path):
    raw,args=_prepare(tmp_path)
    m51_path=args["m51_path"]
    body=json.loads(m51_path.read_text())
    body["videos_byte_identical_total"]=39
    m51_path.write_text(json.dumps(body))
    called=[0]
    def pt(video):
        called[0]+=1
        return [i/25 for i in range(1000)]
    args["pts_probe"]=pt
    with pytest.raises(ValueError,match="previous M51"):
        m52.run(**args)
    assert called==[0]
    m51_path.write_text(json.dumps({
        **body,"videos_byte_identical_total":40
    }))
    args["csv_reader"]=lambda archive,split:raw+b"newline-corruption"
    with pytest.raises(ValueError,match="TRAIN CSV SHA256"):
        m52.run(**args)
    assert called==[0]
    assert not args["report"].exists()


def test_m52_flags_out_of_range_GT_and_refuses_bad_pretest_gate(tmp_path):
    raw,args=_prepare(tmp_path)
    args["evidence_gate"]=lambda **kw:{
        "research_source_provenance_check_passed":False
    }
    with pytest.raises(ValueError,match="source gate failed"):
        m52.run(**args)
    args["evidence_gate"]=lambda **kw:{
        "research_source_provenance_check_passed":True
    }
    args["fps_probe"]=lambda path:(25.,15)
    args["pts_probe"]=lambda path:[k/25 for k in range(15)]
    with pytest.raises(ValueError,match="GT index exceeds"):
        m52.run(**args)
    assert not args["report"].exists()
