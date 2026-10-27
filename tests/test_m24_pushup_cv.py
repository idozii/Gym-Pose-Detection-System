"""M24: synthetic isolated-subject CV, fair identical eval cohort, no TEST."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from gym_pose import gym7_m24_cv as m24


def make_samples(n_people=10):
    rng=np.random.default_rng(100)
    original, full, upper = [], [], []
    for subject in range(n_people):
        subject_id=f"research_person_{subject:02d}"
        for view in ("front","side","diagonal"):
            for label in ("good","bad"):
                video_id=f"video_{subject}_{view}_{label}"
                original.append({
                    "video_id":video_id,
                    "subject_id":subject_id,
                    "archive_member":f"fake/{video_id}.mp4",
                    "exercise":"pushup","view":view,"label":label,
                })
                row={key: original[-1][key] for key in m24.META}
                features=rng.normal(.8 if label=="bad" else -.8,1,64)
                row.update(zip(m24.FEATURE_COLUMNS,map(float,features)))
                upper.append(dict(row))
                if view!="front" or label=="good":
                    full.append(dict(row))
    def table(rows):
        return pd.DataFrame(rows,columns=m24.COLS).set_index("video_id",drop=False)
    return table(full),table(upper),original


def test_m24_paired_cv_has_same_video_cohort_and_no_model_artifact(monkeypatch):
    full,upper,source=make_samples(n_people=10)
    subjects={r["subject_id"] for r in source}
    monkeypatch.setattr(m24,"frozen_train_manifest",
                        lambda *_a,**_k:(source,subjects))
    result=m24.compare_cv(full,upper,source,pd.DataFrame(),folds=5,trees=25)
    assert result["test_accessed"] is False
    assert result["heldout_m20_val_or_test_scored"] is False
    assert result["original_train_pushup_clips"]==60
    assert result["full_gate_clips"]==50
    assert result["upper_gate_clips"]==60
    assert result["newly_accepted_train_clips"]==10
    assert result["paired_common_clips"]==50
    assert set(result["metrics_on_identical_paired_clips"])==set(m24.MODELS)
    assert all(result["metrics_on_identical_paired_clips"][m][
        "pooled_group_oof"]["n"]==50 for m in m24.MODELS)
    assert result["recovered_unpaired_oof_only"]["all_views"]["n"]==10
    assert result["recovered_unpaired_oof_only"]["by_view"]["front"]["n"]==10
    # Aggregate output should NEVER include subjects or source paths.
    dump=json.dumps(result)
    assert "research_person_" not in dump
    assert "fake/" not in dump
    assert "video_1" not in dump


def test_global_m20_train_can_have_curl_only_subject_without_cv_leak(monkeypatch):
    full, upper, source = make_samples(n_people=10)
    pushup_people = {r["subject_id"] for r in source}
    # Actual owner M20 split has 15 global TRAIN people, only 14 Push-up.
    overall_m20_train_people = pushup_people | {"curl_only_train_person"}
    monkeypatch.setattr(
        m24, "frozen_train_manifest",
        lambda *_a, **_k: (source, overall_m20_train_people)
    )
    _, _, eligible, count = m24.verify_cohort(
        full, upper, source, pd.DataFrame()
    )
    assert len(eligible) == 10
    assert "curl_only_train_person" not in eligible
    assert count == 60
    report = m24.compare_cv(
        full, upper, source, pd.DataFrame(), folds=5, trees=25
    )
    assert report["frozen_original_train_subjects"] == 10


def test_m24_rejects_non_train_and_drifted_source_metadata(monkeypatch):
    full,upper,source=make_samples(n_people=10)
    subjects={r["subject_id"] for r in source}
    monkeypatch.setattr(m24,"frozen_train_manifest",
                        lambda *_a,**_k:(source,subjects))
    missing=upper.drop(index=full.index[0])
    with pytest.raises(ValueError,match="Upper admission"):
        m24.verify_cohort(full,missing,source,pd.DataFrame())
    altered=upper.copy()
    altered.loc[full.index[0],"subject_id"]="not_a_train_person"
    with pytest.raises(ValueError,match="metadata differs"):
        m24.verify_cohort(full,altered,source,pd.DataFrame())
    leaked=upper.copy()
    new=leaked.iloc[[0]].copy()
    new["video_id"]="unauthorized_holdout_video"
    new.index=["unauthorized_holdout_video"]
    leaked=pd.concat([leaked,new])
    with pytest.raises(ValueError,match="unknown/non-TRAIN"):
        m24.verify_cohort(full,leaked,source,pd.DataFrame())


def test_m24_table_schema_rejects_missing_or_nonfinite_features(tmp_path):
    full,_,_=make_samples(n_people=10)
    path=tmp_path/"m23_fake.csv"
    full.to_csv(path,index=False)
    assert len(m24._read_table(path))==50
    table=pd.read_csv(path)
    table.loc[0,m24.FEATURE_COLUMNS[0]]=np.nan
    table.to_csv(path,index=False)
    with pytest.raises(ValueError,match="Nonfinite"):
        m24._read_table(path)
    table=table.drop(columns=[m24.FEATURE_COLUMNS[0]])
    table.to_csv(path,index=False)
    with pytest.raises(ValueError,match="64-feature"):
        m24._read_table(path)


def test_m24_invalid_settings_rejected(monkeypatch):
    full,upper,source=make_samples(n_people=10)
    with pytest.raises(ValueError,match="Invalid CV"):
        m24.compare_cv(full,upper,source,pd.DataFrame(),folds=1)
    with pytest.raises(ValueError,match="Invalid CV"):
        m24.compare_cv(full,upper,source,pd.DataFrame(),trees=10)
