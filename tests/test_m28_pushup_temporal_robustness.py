"""M28 synthetic CV, subject isolation and reproducibility regression tests."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from scripts import m28_pushup_temporal_robustness as m28
from scripts.m23_train_only_pushup_features import FEATURE_COLUMNS, META, COLS
from scripts.m27_pushup_temporal_train import MOTION_COLS, ALL_COLS


def synthetic(n_subjects=7):
    rng=np.random.default_rng(7)
    base=[]
    motion=[]
    source=[]
    for person in range(n_subjects):
        for view in ("front","side","diagonal"):
            for label in ("good","bad"):
                vid=f"synthetic_{person}_{label}_{view}"
                meta={"video_id":vid,"subject_id":f"person_{person}",
                      "exercise":"pushup","label":label,"view":view}
                feature={key:float(value) for key,value in zip(
                    FEATURE_COLUMNS,
                    rng.normal(loc=.1 if label=="bad" else -.1,size=64)
                )}
                base.append({**meta,**feature})
                artifacts={key:float(rng.normal()) for key in MOTION_COLS}
                motion.append({**meta,**artifacts})
                source.append({**meta,"archive_member":f"fake/{vid}.mp4"})
    b=pd.DataFrame(base,columns=COLS).set_index("video_id",drop=False)
    m=pd.DataFrame(motion,columns=ALL_COLS).set_index("video_id",drop=False)
    return b,m,source


def test_m28_feature_groups_are_frozen_and_disjoint():
    assert len(m28.FEATURE_SETS["baseline_64"])==64
    assert len(m28.FEATURE_SETS["kinematic_76"])==76
    assert len(m28.FEATURE_SETS["quality_proxy_68"])==68
    assert len(m28.FEATURE_SETS["temporal_80"])==80
    assert set(m28.KINEMATIC).isdisjoint(set(m28.QUALITY))
    assert tuple(m28.KINEMATIC+m28.QUALITY)==tuple(MOTION_COLS)


def test_m28_all_folds_use_separate_subjects(monkeypatch):
    baseline,motion,source=synthetic()
    people={r["subject_id"] for r in source}
    monkeypatch.setattr(m28,"_read_table",lambda _:baseline)
    monkeypatch.setattr(m28,"_read_temporal",lambda _:motion)
    monkeypatch.setattr(m28,"read_manifest",lambda _:source)
    monkeypatch.setattr(m28,"load_clip_table",lambda _:pd.DataFrame())
    monkeypatch.setattr(m28,"frozen_train_manifest",
                        lambda *_args,**_kwargs:(source,people))
    calls=[]
    def fake_fit_and_predict(frame,names,train_idx,held_idx):
        train=set(frame.iloc[train_idx].subject_id)
        held=set(frame.iloc[held_idx].subject_id)
        assert not train&held
        assert train|held==people
        calls.append((len(train),len(held),len(names)))
        return np.asarray([.6 if x=="bad" else .4
                           for x in frame.iloc[held_idx].label])
    monkeypatch.setattr(m28,"_train_predict",fake_fit_and_predict)
    result=m28.run_study("b.csv","m.csv","manifest.csv","source.csv",
                         bootstrap=100)
    assert result["original_M20_VAL_or_TEST_used"] is False
    assert result["raw_video_decoded"] is False
    assert result["paired_train_clip_count"]==42
    assert result["train_people"]==7
    assert result["results"]["group5"]["folds"]==5
    assert result["results"]["loso"]["folds"]==7
    assert len(calls)==(5+7)*4
    assert all(result["results"][strategy]["models"][model][
        "all_paired"]["n"]==42 for strategy in ("group5","loso")
        for model in m28.FEATURE_SETS)
    assert all(result["results"][strategy]["models"][model][
        "all_paired"]["balanced_accuracy"]==1. for strategy in ("group5","loso")
        for model in m28.FEATURE_SETS)
    output=json.dumps(result)
    assert "person_0" not in output
    assert "fake/" not in output


def test_m28_rejects_non_train_or_mismatched_video(monkeypatch):
    baseline,motion,source=synthetic()
    people={r["subject_id"] for r in source}
    monkeypatch.setattr(m28,"frozen_train_manifest",
                        lambda *_args,**_kwargs:(source,people))
    with pytest.raises(ValueError,match="identical original TRAIN"):
        m28.verify_exact_train_cohort(
            baseline, motion.drop(index=motion.index[0]), source,
            pd.DataFrame()
        )
    leak=motion.copy()
    leak.loc[leak.index[0],"subject_id"]="not_in_train"
    with pytest.raises(ValueError,match="subject metadata mismatch"):
        m28.verify_exact_train_cohort(baseline,leak,source,pd.DataFrame())
    leak=motion.copy()
    wrong=leak.iloc[[0]].copy()
    wrong.index=["TEST_video"]
    wrong.loc[:,"video_id"]="TEST_video"
    with pytest.raises(ValueError,match="identical original TRAIN"):
        m28.verify_exact_train_cohort(
            baseline,pd.concat([motion,wrong]),source,pd.DataFrame()
        )


def test_paired_subject_bootstrap_remains_zero_for_identical_predictions():
    truth={str(i):int(i%2==1) for i in range(12)}
    groups={str(i):f"person_{i//2}" for i in range(12)}
    scores={name:{i:(.8 if y else .2) for i,y in truth.items()}
            for name in m28.FEATURE_SETS}
    report=m28.paired_cluster_bootstrap(
        truth,scores,groups,resamples=100,seed=42
    )
    for variant in m28.PAIRED_ALTERNATIVES:
        for interval in report["paired_deltas"][variant].values():
            assert interval["delta_vs_baseline_64_lower_95"]==0
            assert interval["delta_vs_baseline_64_upper_95"]==0
    with pytest.raises(ValueError,match="100..5000"):
        m28.paired_cluster_bootstrap(truth,scores,groups,resamples=10)
    mismatched={k:dict(v) for k,v in scores.items()}
    mismatched["temporal_80"].pop("0")
    with pytest.raises(ValueError,match="same paired cohort"):
        m28.paired_cluster_bootstrap(truth,mismatched,groups,resamples=100)


def test_m27_crosscheck_refuses_metric_drift(tmp_path):
    d={"results":{"group5":{"models":{}}}}
    old={"metrics":{}}
    for name in ("baseline_64","temporal_80"):
        metrics={k:.5 for k in (
            "balanced_accuracy","f1_bad","recall_bad","average_precision_bad"
        )}
        d["results"]["group5"]["models"][name]={"all_paired":dict(metrics)}
        old["metrics"][name]={"all_paired":dict(metrics)}
    path=tmp_path/"m27_report.json"
    path.write_text(json.dumps(old))
    assert "reproduced exactly" in m28._check_m27_repro(d,path)
    old["metrics"]["temporal_80"]["all_paired"]["balanced_accuracy"]=.6
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError,match="failed to reproduce"):
        m28._check_m27_repro(d,path)
