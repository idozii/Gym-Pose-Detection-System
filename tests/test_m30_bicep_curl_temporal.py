"""M30 Curl experiments use artificial metadata and no actual exercise media."""
from __future__ import annotations

import csv
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts import m30_bicep_curl_temporal as m30


def fake_pose():
    p=[SimpleNamespace(x=.1 + .002*i,y=.2+.001*i,z=0,visibility=.95)
       for i in range(33)]
    for i in (11,12):p[i].x,p[i].y=.45,.2
    for i in (23,24):p[i].x,p[i].y=.45,.6
    p[13].x,p[13].y=.35,.3
    p[15].x,p[15].y=.3,.47
    p[14].x,p[14].y=.55,.3
    p[16].x,p[16].y=.60,.47
    return p


def synthetic():
    rng=np.random.default_rng(9)
    original=[]
    rows=[]
    for subject in range(9):
        for view in ("front","side","diagonal"):
            for label in ("good","bad"):
                video=f"curl_{subject}_{view}_{label}"
                meta={"video_id":video,"subject_id":f"person_{subject}",
                      "exercise":"bicep_curl","label":label,"view":view}
                src={**meta,"archive_member":f"{video}.mp4"}
                rows.append(src)
                record={**meta,"archive_member":src["archive_member"]}
                record.update(zip(m30.FEATURES,rng.normal(
                    0 if label=="good" else .4,1,len(m30.FEATURES)
                )))
                original.append(record)
    return rows,pd.DataFrame(original,columns=(*m30.META,"archive_member",*m30.FEATURES))


def fake_splits(monkeypatch):
    monkeypatch.setattr(m30,"split_by_subject",lambda frame,**_kw: {
        "train":frame[frame.subject_id.isin({f"person_{i}" for i in range(7)})],
        "val":frame[frame.subject_id=="person_7"],
        "test":frame[frame.subject_id=="person_8"]
    })


def motion_table(ids,source):
    mapping={r["video_id"]:r for r in source}
    records=[]
    for pos,vid in enumerate(ids):
        src=mapping[vid]
        entry={k:src[k] for k in m30.META}
        entry.update({key:float(pos/10+i/16) for i,key
                      in enumerate(m30.MOTION_COLS)})
        records.append(entry)
    return pd.DataFrame(records,columns=m30.MOTION_FILE_COLS)


def test_m30_curl_pose_motion_is_not_a_form_label():
    sample=fake_pose()
    result=m30.curl_motion_frame(sample)
    assert result is not None
    assert 0<=result[0]<=180
    assert np.isfinite(result[1])
    assert m30.curl_motion_frame(None) is None
    sample[11].visibility=0
    sample[12].visibility=0
    assert m30.curl_motion_frame(sample) is None


def test_m30_strict_original_subject_disjoint_selection(monkeypatch):
    source,original=synthetic()
    fake_splits(monkeypatch)
    ids,train,mapping=m30.frozen_curl_train(source,original)
    assert len(ids)==42 and train.subject_id.nunique()==7
    assert not any("person_8"==mapping[vid]["subject_id"] for vid in ids)
    altered=[dict(r) for r in source]
    altered[0]["subject_id"]="person_8"
    with pytest.raises(ValueError,match="subject, view or label mismatch"):
        m30.frozen_curl_train(altered,original)
    with pytest.raises(ValueError,match="Duplicate source"):
        m30.frozen_curl_train(source+[source[0]],original)


def test_m30_extract_never_decodes_val_or_test_video(tmp_path,monkeypatch):
    source,original=synthetic()
    fake_splits(monkeypatch)
    monkeypatch.setattr(m30,"read_manifest",lambda _path:source)
    monkeypatch.setattr(m30,"load_clip_table",lambda _path:original)
    class Dummy:
        def __enter__(self):return self
        def __exit__(self,*_args):return False
    fake_mp=SimpleNamespace(solutions=SimpleNamespace(
        pose=SimpleNamespace(Pose=lambda **_kw:Dummy())
    ))
    monkeypatch.setitem(sys.modules,"mediapipe",fake_mp)
    monkeypatch.setitem(sys.modules,"cv2",SimpleNamespace())
    seen=[]
    def extractor(_zip,entry,*_args,**_kw):
        seen.append(entry.filename)
        return np.ones(len(m30.MOTION_COLS)),None
    monkeypatch.setattr(m30,"_extract_video",extractor)
    archive=tmp_path/"synthetic.zip"
    with zipfile.ZipFile(archive,"w") as zipped:
        for row in source:zipped.writestr(row["archive_member"],b"fake")
    output=tmp_path/"motion.csv"
    report=m30.extract(archive,tmp_path/"manifest",
                       tmp_path/"original",output)
    assert report["original_train_subjects_for_curl"]==7
    assert report["original_train_curl_clips"]==42
    assert report["original_VAL_or_TEST_decoded"] is False
    assert len(seen)==42
    assert not any("curl_7_" in item or "curl_8_" in item for item in seen)
    assert len(pd.read_csv(output))==42
    with pytest.raises(FileExistsError):
        m30.extract(archive,tmp_path/"manifest",tmp_path/"original",output)


def test_m30_grouped_cv_uses_exact_same_train_clips(tmp_path,monkeypatch):
    source,original=synthetic()
    fake_splits(monkeypatch)
    monkeypatch.setattr(m30,"read_manifest",lambda _path:source)
    monkeypatch.setattr(m30,"load_clip_table",lambda _path:original)
    ids,_,_=m30.frozen_curl_train(source,original)
    motion=motion_table(ids,source)
    path=tmp_path/"motion.csv"
    motion.to_csv(path,index=False)
    monkeypatch.setattr(m30,"TREES",25)
    result=m30.compare(path,tmp_path/"manifest",tmp_path/"original")
    assert result["subjects"]==7
    assert result["paired_clips"]==42
    assert result["models"]["group5"]["folds"]==5
    assert result["models"]["loso"]["folds"]==7
    assert set(result["feature_sets"])==set(m30.FAMILIES)
    assert all(result["models"][strategy]["models"][name][
        "all_paired"]["n"]==42 for strategy in ("group5","loso")
        for name in m30.FAMILIES)
    assert result["original_M20_VAL_or_TEST_scored"] is False
    corrupt=motion.iloc[:-1]
    corrupt.to_csv(path,index=False)
    with pytest.raises(ValueError,match="cohorts differ"):
        m30.compare(path,tmp_path/"manifest",tmp_path/"original")
    injected=motion.copy()
    injected.loc[0,"subject_id"]="person_8"
    injected.to_csv(path,index=False)
    with pytest.raises(ValueError,match="metadata mismatch"):
        m30.compare(path,tmp_path/"manifest",tmp_path/"original")
