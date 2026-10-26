"""Synthetic OHP benchmark fixtures, no publisher archive, test, MediaPipe or network."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gym_pose.ohp_task_features import FEATURES_BY_TASK, SCHEMA_BY_TASK, TASKS
from scripts import benchmark_fitness_aqa_ohp_tasks as bench


def synthetic_source(tmp_path: Path, *, pilot=False, leak=False):
    root = tmp_path / "features"
    root.mkdir()
    rng = np.random.default_rng(7)
    summary = {
        "status": "OHP_task_specific_pose_extraction_NO_TEST",
        "source_split_type": "publisher_video_ID_disjoint_NOT_subject_disjoint",
        "schema_by_task": SCHEMA_BY_TASK,
        "feature_count_by_task": {t: len(FEATURES_BY_TASK[t]) for t in TASKS},
        "source_test_videos_processed": 0,
        "sampling_fps": 2.,
        "boundary_margin_seconds": .1,
        "min_visibility_on_required_joints": .5,
        "limit_per_split": 8 if pilot else 0,
        "splits": {},
    }
    for split in ("train", "val"):
        video_count = 48 if split=="train" else 16
        vids = [f"{1000+i}_{1}" if split=="train" else f"{2000+i}_{1}"
                for i in range(video_count)]
        if leak and split=="val":
            vids[0] = "1000_1"
        task_stats = {}
        for task in TASKS:
            records = []
            for n, vid in enumerate(vids):
                for frame in range(4):
                    # Simulate different task admission with separate sample universes.
                    if task=="error_elbows" and (n+frame)%9==0:
                        continue
                    if task=="error_knees" and (n+frame)%7==0:
                        continue
                    ts = frame / 2.
                    positive_elbow = int(n%2==0 and frame>=1)
                    positive_knee = int(n%3!=0 and frame>=1)
                    features = {
                        name: float(rng.normal(.1, .02) + (.1 if (
                            positive_elbow if task=="error_elbows" else positive_knee
                        ) else 0.)) for name in FEATURES_BY_TASK[task]
                    }
                    records.append({
                        "sample_id": f"{vid}_f{frame:07d}",
                        "video_id": vid, "split": split,
                        "timestamp_s": ts, "frame_index": frame,
                        "source_fps": 30., "error_elbows": positive_elbow,
                        "error_knees": positive_knee,
                        "any_of_two_annotated_errors": int(
                            positive_elbow or positive_knee),
                        **features,
                    })
            df = pd.DataFrame(records, columns=[
                *bench.META_COLUMNS, *FEATURES_BY_TASK[task]])
            df.to_csv(root / f"fitness_aqa_ohp_{task}_{split}_pose.csv",
                      index=False)
            task_stats[task] = {
                "accepted_frames": len(df),
                "rejected_frames": video_count*4 - len(df),
                "positive_frames": int(df[task].sum()),
                "negative_frames": int(len(df)-df[task].sum()),
                "videos_with_accepted_frames": int(df.video_id.nunique()),
                "videos_with_positive_frames": int(
                    df.loc[df[task].eq(1), "video_id"].nunique()),
                "pose_coverage_eligible": len(df)/(video_count*4),
            }
        summary["splits"][split] = {
            "selected_videos": video_count, "videos_processed": video_count,
            "videos_failed": 0, "eligible_frames": video_count*4,
            "tasks": task_stats,
        }
    (root / "fitness_aqa_ohp_task_extraction_summary.json").write_text(
        json.dumps(summary))
    # A corrupt TEST CSV should never be touched.
    (root / "fitness_aqa_ohp_error_elbows_test_pose.csv").write_text(
        "THIS FILE MUST NEVER BE READ")
    return root


def test_temporal_history_is_strictly_causal_and_resets_after_gap():
    task="error_elbows"
    cols=list(FEATURES_BY_TASK[task])
    rows=[]
    for idx, value in enumerate((1., 2., 99., 11.)):
        row={key:.1 for key in cols}
        row[cols[-6]]=value
        row["video_id"]="123_1" if idx<3 else "321_1"
        row["timestamp_s"]=(0., .5, 2., 0.)[idx]
        row["frame_index"]=(0, 5, 20, 0)[idx]
        row["sample_id"]=f"{row['video_id']}_f{row['frame_index']:07d}"
        rows.append(row)
    df=pd.DataFrame(rows).iloc[[2,0,3,1]].reset_index(drop=True)
    result=bench.task_features(df,task,temporal=True,max_gap_seconds=.75)
    assert result.shape==(4,42)
    assert bench.task_features(df,task,temporal=False).shape==(4,30)
    signal=cols[-6]
    by_id=pd.Series(result[f"{signal}_delta1"].values,index=df.sample_id)
    assert by_id.loc["123_1_f0000000"] == 0
    assert by_id.loc["123_1_f0000005"] == 1
    assert by_id.loc["123_1_f0000020"] == 0  # >.75s gap
    assert by_id.loc["321_1_f0000000"] == 0  # different source clip
    changed=df.copy()
    changed.loc[changed.sample_id.eq("123_1_f0000020"),signal]=999.
    after=bench.task_features(changed,task,temporal=True,max_gap_seconds=.75)
    base_ids=~df.sample_id.eq("123_1_f0000020")
    np.testing.assert_allclose(result.loc[base_ids],after.loc[base_ids])


def test_group_calibration_split_is_disjoint_and_has_annotated_videos(tmp_path):
    root=synthetic_source(tmp_path)
    frames,_=bench._load_data(root,require_full=False)
    for task in TASKS:
        train=frames[task]["train"]
        fit,cal=bench.group_calibration(train,task=task,seed=42)
        assert not set(train.iloc[fit].video_id)&set(train.iloc[cal].video_id)
        assert train.iloc[fit].loc[train.iloc[fit][task].eq(1),"video_id"].nunique()>=10
        assert train.iloc[cal].loc[train.iloc[cal][task].eq(1),"video_id"].nunique()>=3


def test_paired_bootstrap_identical_models_have_zero_difference():
    ids=np.array(["a"]*4+["b"]*4+["c"]*4+["d"]*4)
    y=np.array([0,0,0,0, 0,1,1,1, 1,1,1,0, 0,0,0,0])
    p=np.array([.1,.2,.2,.1, .1,.7,.8,.7, .8,.9,.8,.2, .2,.3,.2,.1])
    data=bench.paired_video_bootstrap(
        y,ids,{"static":p,"causal_temporal":p},
        {"static":.5,"causal_temporal":.5},repetitions=100,seed=123)
    for metric in ("ap","macro_f1"):
        ci=data["intervals"]["temporal_minus_static"][metric]
        assert ci["percentile_2_5"]==0
        assert ci["percentile_97_5"]==0
    assert data["valid_resamples"]+data["skipped_single_class_resamples"]==100


def test_benchmark_end_to_end_saves_only_trusted_train_fits(tmp_path):
    root=synthetic_source(tmp_path)
    output=tmp_path/"model_artifacts"
    report=bench.benchmark(root,output,seed=42,trees=16,
                           bootstraps=100,require_full=False)
    assert report["test_data_accessed"] is False
    assert report["validation_used_for_threshold_selection"] is False
    assert set(report["tasks"])==set(TASKS)
    assert (output/"comparison.json").is_file()
    for task in TASKS:
        result=report["tasks"][task]
        assert result["calibration"]["disjoint"]
        assert result["validation_positive_videos"]>=5
        assert result["paired_validation_cluster_bootstrap"]["paired_same_frame_population"]
        assert set(result["variants"])==set(bench.VARIANTS)
        assert result["variants"]["static"]["features_count"]==30
        assert result["variants"]["causal_temporal"]["features_count"]==42
        for variant in bench.VARIANTS:
            path=output/f"{task}_{variant}.joblib"
            assert path.is_file()
            threshold=result["variants"][variant]["train_only_threshold"]
            assert .05<=threshold<=.95


def test_pilot_and_stale_extract_summary_are_blocked(tmp_path):
    root=synthetic_source(tmp_path,pilot=True)
    with pytest.raises(ValueError,match="Pilot"):
        bench._load_data(root,require_full=True)
    frames,_=bench._load_data(root,require_full=False)
    assert len(frames["error_elbows"]["train"])>0
    summary_path=root/"fitness_aqa_ohp_task_extraction_summary.json"
    summary=json.loads(summary_path.read_text())
    summary["splits"]["train"]["tasks"]["error_elbows"]["positive_frames"]+=1
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="summary and CSV disagree"):
        bench._load_data(root,require_full=False)


def test_video_id_leakage_blocks_benchmark(tmp_path):
    root=synthetic_source(tmp_path,leak=True)
    with pytest.raises(ValueError,match="leakage"):
        bench._load_data(root,require_full=False)


def test_invalid_source_csv_rejected(tmp_path):
    root=synthetic_source(tmp_path)
    path=root/"fitness_aqa_ohp_error_knees_val_pose.csv"
    df=pd.read_csv(path)
    df.loc[0,FEATURES_BY_TASK["error_knees"][0]]=float("nan")
    df.to_csv(path,index=False)
    with pytest.raises(ValueError,match="missing feature/metadata"):
        bench._load_data(root,require_full=False)


def test_bootstrap_and_model_parameters_checked():
    with pytest.raises(ValueError,match="Need >=100"):
        bench.paired_video_bootstrap(
            np.array([0,1]),np.array(["a","b"]),
            {"static":np.array([.2,.8]),"causal_temporal":np.array([.2,.8])},
            {"static":.5,"causal_temporal":.5},
            repetitions=5)
    with pytest.raises(ValueError,match="trees"):
        bench.model(seed=1,trees=0)
