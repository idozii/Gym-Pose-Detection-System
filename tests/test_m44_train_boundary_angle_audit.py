"""M44 synthetic TRAIN fixture tests. Never read actual publisher videos."""
from __future__ import annotations

import csv
import io
import json
import tarfile

import pytest

from scripts.m44_train_boundary_angle_audit import run, PROTOCOL
from scripts.m44_boundary_angle_schema import (
    ANGLE_BIN_KEYS, CONTEXT_KEYS, FP_LOCATION_KEYS,
    validate_m44, validate_against_m43
)


def _add(tf, name, contents):
    entry=tarfile.TarInfo(name)
    entry.size=len(contents)
    tf.addfile(entry,io.BytesIO(contents))


def archive(tmp_path):
    p=tmp_path/"RepCountA.tar.gz"
    csv_file=io.StringIO()
    writer=csv.DictWriter(csv_file,fieldnames=["name","type","count","L1","L2"])
    writer.writeheader()
    writer.writerow({"name":"source_squat.mp4","type":"squat",
                     "count":1,"L1":0,"L2":10})
    rows=csv_file.getvalue().encode("utf8")
    with tarfile.open(p,"w:gz") as tf:
        _add(tf,"LLSP/annotation/train.csv",rows)
        _add(tf,"LLSP/annotation/valid.csv",rows)
        _add(tf,"LLSP/video/train/source_squat.mp4",b"r"*15000)
        _add(tf,"LLSP/video/valid/source_squat.mp4",b"v"*15000)
        _add(tf,"LLSP/video/test/source_squat.mp4",b"t"*15000)
    return p


def _base_method(n, count, tp):
    return {
        "annotated_count":n, "predicted_count":count,
        "absolute_count_error":abs(count-n),
        "signed_count_error":count-n,
        "event_tp":tp, "event_fp":count-tp,"event_fn":n-tp
    }


def _fake_replay(source, target):
    trace=json.loads(source.read_text())
    n=trace["annotated_count"]
    def condition(samples,tp):
        return {
            "protocol":"m39-train-only-temporal-return-v1",
            "source_split":"train","annotation_rep_count":n,
            "input_frame_samples":samples,"pose_sample_coverage":0.9,
            "m34":_base_method(n,0,0),
            "always_zero":_base_method(n,0,0),
            "low_to_high":_base_method(n,tp,tp)
        }
    m41={
        "protocol":"m41-train-cadence-sensitivity-v1",
        "source_split":"train","annotated_rep_count":n,
        "dense_200ms":condition(20,0),
        "subsample_400ms":condition(10,0),
        "design":{
            "nominal_dense_ms":200,"downsample_factor":2,
            "shared_python_pose_tracking_context":True,
            "m34_internal_sample_gate_unchanged_ms":400,
            "exact_independent_400ms_pose_estimator_parity":False
        }
    }
    def diagnostic():
        return {
            "gt_events":n,"tp":0,"fp":0,"fn":n,
            "unmatched_gt_window_observations":{
                "no_usable_angle_in_gt_window":0,
                "usable_but_no_low_in_gt_window":n,
                "low_but_no_later_high_in_gt_window":0,
                "low_then_high_in_gt_window_still_unmatched":0
            },
            "unmatched_predictions_by_tolerance":{
                "unmatched_within_some_gt_end_tolerance":0,
                "unmatched_outside_all_gt_end_tolerances":0
            },
            "all_gt_windows_usable_lt4":n,"all_gt_windows_usable_lt6":n,
            "all_gt_windows_no_usable":0,
            "all_gt_windows_with_any_low":0,
            "all_gt_windows_with_low_then_high":0,
            "side_switch_resets_while_pending":0,
            "tracking_gap_resets_while_pending":0
        }
    m43={
        "protocol":"m43-train-temporal-error-audit-v1",
        "source_split":"train","annotated_rep_count":n,
        "m41":m41,
        "diagnostics":{
            "dense_200ms":diagnostic(),
            "subsample_400ms":diagnostic(),
            "paired_gt_outcomes":{
                "both_matched":0,"dense_only_matched":0,
                "subsample_only_matched":0,"neither_matched":n
            }
        }
    }
    def angles():
        return {
            "unmatched_gt_no_low_inside":n,
            "min_angle_bins_for_no_low_gt":{
                key: n if key=="min_gt_angle_gt135_to150" else 0
                for key in ANGLE_BIN_KEYS
            },
            "low_in_750ms_context_for_no_low_gt":{
                key: n if key=="low_neither_side" else 0
                for key in CONTEXT_KEYS
            },
            "no_low_gt_with_usable_angle_before":n,
            "no_low_gt_with_usable_angle_after":n,
            "unmatched_prediction_location":{key:0 for key in FP_LOCATION_KEYS}
        }
    result={
        "protocol":PROTOCOL,"source_split":"train",
        "annotated_rep_count":n,"context_ms":750,"m43":m43,
        "conditions":{"dense_200ms":angles(),"subsample_400ms":angles()}
    }
    target.write_text(json.dumps(result),encoding="utf8")
    return result


def test_m44_real_pipeline_contract_without_real_media(tmp_path):
    source_archive=archive(tmp_path)
    called=[]
    def extract(video,ann,*,sample_ms):
        called.append((video.read_bytes()[:1],ann["annotated_count"],sample_ms))
        return {
            "annotated_count":ann["annotated_count"],
            "frames":[{"time_ms":i*200,"landmarks":None} for i in range(20)]
        }
    r=run(
        archive=source_archive,root=tmp_path/"local",
        report_path=tmp_path/"report.json",limit=1,
        extract_fn=extract,compare_fn=_fake_replay
    )
    assert called==[(b"r",1,200)]
    assert r["videos_evaluated"]==1 and r["videos_excluded"]==0
    assert r["annotated_reps"]==1
    assert r["m44_boundary_diagnostics"]["dense_200ms"]["unmatched_gt_no_low_inside"]==1
    assert r["m44_boundary_diagnostics"]["dense_200ms"][
        "min_angle_bins_for_no_low_gt"]["min_gt_angle_gt135_to150"]==1
    assert r["validation_or_test_accessed"] is False
    assert r["original_detectors_modified"] is False
    raw=(tmp_path/"report.json").read_text()
    assert "source_squat" not in raw
    assert '"landmarks"' not in raw
    assert '"frames"' not in raw
    assert '"time_ms"' not in raw
    assert '"cycle_bounds_frames"' not in raw
    assert not list((tmp_path/"local").rglob("valid/*.mp4"))
    assert not list((tmp_path/"local").rglob("test/*.mp4"))
    assert not list((tmp_path/"local").glob("m44-private-*"))
    with pytest.raises(FileExistsError,match="refusing overwrite"):
        run(archive=source_archive,root=tmp_path/"local",
            report_path=tmp_path/"report.json",limit=1)


def test_m44_strict_schema_rejects_false_partition_and_bad_original_counts(tmp_path):
    p=tmp_path/"input.json"
    p.write_text('{"annotated_count":1}')
    record=_fake_replay(p,tmp_path/"output.json")
    validate_m44(record["conditions"])
    validate_against_m43(record["conditions"],record["m43"]["diagnostics"])
    broken=json.loads(json.dumps(record))
    broken["conditions"]["dense_200ms"][
        "min_angle_bins_for_no_low_gt"]["min_gt_angle_gt135_to150"]=0
    with pytest.raises(ValueError,match="partition"):
        validate_m44(broken["conditions"])
    broken=json.loads(json.dumps(record))
    broken["conditions"]["subsample_400ms"]["unmatched_gt_no_low_inside"]=2
    with pytest.raises(ValueError,match="partition"):
        validate_m44(broken["conditions"])
    broken=json.loads(json.dumps(record))
    broken["m43"]["diagnostics"]["dense_200ms"][
        "unmatched_gt_window_observations"]["usable_but_no_low_in_gt_window"]=0
    with pytest.raises(ValueError,match="frozen M43"):
        validate_against_m43(broken["conditions"],broken["m43"]["diagnostics"])


def test_m44_blocks_bad_limits_and_unreadable_archives(tmp_path):
    with pytest.raises(ValueError,match="1..20"):
        run(archive=tmp_path/"missing.tar.gz",root=tmp_path/"r",
            report_path=tmp_path/"r.json",limit=21)
    with pytest.raises(ValueError,match="owner-local"):
        run(archive=tmp_path/"missing.tar.gz",root=tmp_path/"r",
            report_path=tmp_path/"r.json",limit=5)
