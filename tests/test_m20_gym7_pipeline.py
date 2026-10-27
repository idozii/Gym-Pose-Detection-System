"""M20 reproducibility tests: synthetic metadata, features and RF only."""
from __future__ import annotations
import csv
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from gym_pose.gym7_research import (
    META, FEATURES, EXERCISES, infer_metadata,scan_zip,
    read_manifest, frame_pose_vector, clip_aggregate,
)
from gym_pose.gym7_training import split_by_subject, train_from_clip_csv
from scripts.download_m20_gym7 import select_official_file


def test_choose_single_official_zip_never_all_three():
    meta={"version":3,"files":[
      {"id":"abc","filename":"Dataset Subject-wise.zip"},
      {"id":"def","filename":"Dataset Exercise-wise.zip"},
      {"id":"ghi","filename":"Dataset Exercise Quality-wise.zip"},
    ]}
    assert select_official_file(meta)["id"]=="def"
    with pytest.raises(ValueError,match="version mismatch"):
        select_official_file({**meta,"version":2})
    with pytest.raises(ValueError,match="Exact"):
        select_official_file({"version":3,"files":[]})


def test_scan_produces_explicit_metadata_no_invented_subjects(tmp_path):
    archive=tmp_path/"videos.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("dataset/subject_01/push_up/good/front/a.mp4",b"fake")
        z.writestr("dataset/subject_02/bicep_curl/bad/side/b.mp4",b"fake")
        z.writestr("dataset/no_subject/push_up/diagonal/c.mp4",b"fake")
        z.writestr("dataset/subject_02/squat/good/d.mp4",b"fake")
    manifest=tmp_path/"manifest.csv"
    report=scan_zip(archive,manifest)
    assert report["videos"]==3
    assert report["metadata_rows_needing_review"]==1
    with manifest.open() as f: rows=list(csv.DictReader(f))
    assert rows[0]["subject_id"]==""
    assert any(r["exercise"]=="bicep_curl" and r["label"]=="bad" for r in rows)
    with pytest.raises(ValueError,match="Missing"):
        read_manifest(manifest)
    with pytest.raises(FileExistsError):
        scan_zip(archive,manifest)


def test_reject_traversal_duplicate_and_fake_subject(tmp_path):
    record={"video_id":"v1","archive_member":"../bad/push_up.mp4",
      "subject_id":"subject_001","exercise":"pushup","label":"good","view":"front"}
    p=tmp_path/"manifest.csv"
    pd.DataFrame([record],columns=META).to_csv(p,index=False)
    with pytest.raises(ValueError,match="Unsafe"):
        read_manifest(p)
    record["archive_member"]="a.mp4"
    pd.DataFrame([record,record],columns=META).to_csv(p,index=False)
    with pytest.raises(ValueError,match="Duplicate"):
        read_manifest(p)


def _landmarks():
    return [SimpleNamespace(x=.5+i*.002,y=.4+(i%9)*.03,z=.1,visibility=.99)
            for i in range(33)]


def test_clip_temporal_aggregate_has_264_finite_features():
    a=frame_pose_vector(_landmarks(),"pushup")
    b=frame_pose_vector(_landmarks(),"bicep_curl")
    assert a is not None and a.shape==(132,) and b.shape==(132,)
    assert frame_pose_vector(_landmarks()[:20],"pushup") is None
    assert clip_aggregate([a,a,a]) is None
    sample=clip_aggregate([a,b,a,b])
    assert sample.shape==(264,) and np.isfinite(sample).all()


def _synthetic_table(subjects=24):
    rows=[]
    rng=np.random.default_rng(31)
    for subject in range(subjects):
        for exercise in EXERCISES:
            for label in ("good","bad"):
                row={"video_id":f"{subject}_{exercise}_{label}",
                "archive_member":f"safe/{subject}/{exercise}_{label}.mp4",
                "subject_id":f"subject_{subject:03d}",
                "exercise":exercise,"label":label,
                "view":("front","side","diagonal")[subject%3]}
                features=rng.normal(loc=0 if label=="good" else .2,size=len(FEATURES))
                row.update(zip(FEATURES,features))
                rows.append(row)
    return pd.DataFrame(rows,columns=(*META,*FEATURES))


def test_subject_disjoint_model_baseline_does_not_score_test_by_default(tmp_path):
    path=tmp_path/"features.csv"
    frame=_synthetic_table()
    frame.to_csv(path,index=False)
    splits=split_by_subject(frame)
    groups={k:set(v.subject_id) for k,v in splits.items()}
    assert not groups["train"]&groups["val"]
    assert not groups["train"]&groups["test"]
    assert not groups["val"]&groups["test"]
    result=train_from_clip_csv(path,tmp_path/"output",n_estimators=25)
    assert result["test_accessed"] is False
    assert set(result["models"])==set(EXERCISES)
    assert all("test" not in x["evaluations"] for x in result["models"].values())
    assert (tmp_path/"output"/"pushup_clip_quality_rf.joblib").is_file()
    assert (tmp_path/"output"/"bicep_curl_clip_quality_rf.joblib").is_file()
    with pytest.raises(FileExistsError):
        train_from_clip_csv(path,tmp_path/"output",n_estimators=25)


def test_reject_subject_shortage_and_missing_publisher_labels(tmp_path):
    with pytest.raises(ValueError,match="12 unique"):
        split_by_subject(_synthetic_table(8))
    small=_synthetic_table()
    small.loc[:,"label"]="good"
    path=tmp_path/"bad.csv"
    small.to_csv(path,index=False)
    with pytest.raises(ValueError,match="Not enough"):
        train_from_clip_csv(path,tmp_path/"none",n_estimators=25)
