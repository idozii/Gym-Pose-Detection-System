"""M27 tests operate only on synthetic poses and local in-memory metadata."""
from __future__ import annotations

from types import SimpleNamespace
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts import m27_pushup_temporal_train as m27
from scripts.m23_train_only_pushup_features import (
    FEATURE_COLUMNS, META, COLS
)


def pose(elbow_offset=0):
    lm = [SimpleNamespace(x=.20+i*.001, y=.20+i*.002,
                          z=.01, visibility=.98) for i in range(33)]
    for index in (11, 12):
        lm[index].x, lm[index].y = .45, .20
    for index in (23, 24):
        lm[index].x, lm[index].y = .45, .6
    lm[13].x, lm[13].y = .35, .34
    lm[15].x, lm[15].y = .32 + elbow_offset, .48
    lm[14].x, lm[14].y = .55, .34
    lm[16].x, lm[16].y = .58, .48
    return lm


def test_timeline_features_capture_order_and_rate():
    const = [(i*.5, 90., 1.2, 0) for i in range(8)]
    changing = [(i*.5, 80.+(15 if i%2 else 0), 1.2, 0)
                for i in range(8)]
    a = m27.motion_summary(const, 8)
    b = m27.motion_summary(changing, 8)
    assert len(a) == len(b) == 16
    assert a[5] == 0.
    assert b[5] > a[5]
    assert b[7] > a[7]
    assert m27.motion_summary(changing[:3], 8) is None
    assert m27.motion_summary(changing, 0) is None
    assert m27.motion_summary(changing[:4] + [changing[3]], 9) is None
    with_gap = [(0.,90.,1.,0), (1.,100.,1.,0),
                (5.,105.,1.,0), (6.,115.,1.,1)]
    slow = m27.motion_summary(with_gap, 20)
    assert slow is not None
    assert slow[13] == 4/20
    assert slow[12] == 6.


def test_pose_anatomical_geometry_and_missing_person_are_explicit():
    values = m27.motion_frame(pose())
    assert values is not None
    assert np.isfinite(values[:2]).all()
    assert 0. <= values[0] <= 180.
    p = pose(elbow_offset=.05)
    # One side can be missing; the other is a valid upper-side gate.
    for i in (11, 13, 15, 23):
        p[i].visibility = .1
    assert m27.motion_frame(p) is not None
    assert m27.motion_frame(None) is None
    obscured = pose()
    for i in (11, 12):
        obscured[i].visibility = .01
    assert m27.motion_frame(obscured) is None


def _fake_table(persons=8):
    rows=[]
    rng=np.random.default_rng(2)
    for person in range(persons):
        for quality in ("good", "bad"):
            for camera in ("front", "side"):
                video_id=f"video_{person}_{quality}_{camera}"
                row={"video_id":video_id,"subject_id":f"subject_{person}",
                     "exercise":"pushup","label":quality,"view":camera}
                row.update(zip(FEATURE_COLUMNS, rng.normal(
                    0 if quality=="good" else .4, 1, len(FEATURE_COLUMNS)
                )))
                rows.append(row)
    return pd.DataFrame(rows, columns=COLS).set_index("video_id",drop=False)


def _fake_motion(upper):
    output=upper.loc[:,META].copy()
    for i, feature in enumerate(m27.MOTION_COLS):
        output[feature]=(upper.label=="bad").astype(float)*.1+i/20
    return output


def test_m27_group_cv_paired_same_ids_and_source_train_provenance(monkeypatch,
                                                                   tmp_path):
    upper=_fake_table()
    motion=_fake_motion(upper)
    upper_path=tmp_path/"upper.csv"
    motion_path=tmp_path/"motion.csv"
    upper.to_csv(upper_path,index=False)
    motion.to_csv(motion_path,index=False)
    src=[
        {**{k:getattr(record,k) for k in META},
         "archive_member":f"safe/{record.video_id}.mp4"}
        for record in upper.itertuples(index=False)
    ]
    monkeypatch.setattr(m27,"read_manifest",lambda _:src)
    monkeypatch.setattr(m27,"load_clip_table",lambda _:pd.DataFrame())
    monkeypatch.setattr(
        m27,"frozen_train_manifest",
        lambda source,_original:(src,{r["subject_id"] for r in src})
    )
    summary=m27.compare(motion_path,upper_path,
                        tmp_path/"manifest",tmp_path/"original")
    assert summary["subjects"]==8
    assert summary["paired_clips"]==32
    assert summary["group_kfold_folds"]==5
    assert summary["original_M20_VAL_or_TEST_used"] is False
    assert summary["metrics"]["baseline_64"]["all_paired"]["n"]==32
    assert summary["metrics"]["temporal_80"]["all_paired"]["n"]==32
    assert "subject_0" not in str(summary)
    invalid=_fake_motion(upper.drop(index=upper.index[0]))
    invalid.to_csv(motion_path,index=False)
    with pytest.raises(ValueError,match="identical video IDs"):
        m27.compare(motion_path,upper_path,
                    tmp_path/"manifest",tmp_path/"original")
    motion.to_csv(motion_path,index=False)
    leak=motion.copy()
    leak.iloc[0,leak.columns.get_loc("subject_id")]="test_subject"
    leak.to_csv(motion_path,index=False)
    with pytest.raises(ValueError,match="Source TRAIN metadata mismatch"):
        m27.compare(motion_path,upper_path,
                    tmp_path/"manifest",tmp_path/"original")


def test_m27_extraction_decodes_only_frozen_train_upper_clips(monkeypatch,
                                                               tmp_path):
    upper=_fake_table(persons=2)
    rows=[
        {**{key:getattr(r,key) for key in META},
         "archive_member":f"{r.video_id}.mp4"}
        for r in upper.itertuples(index=False)
    ]
    val={"video_id":"excluded_VAL","subject_id":"VAL_person","exercise":"pushup",
         "label":"bad","view":"front","archive_member":"excluded_VAL.mp4"}
    test={"video_id":"excluded_TEST","subject_id":"TEST_person","exercise":"pushup",
          "label":"good","view":"front","archive_member":"excluded_TEST.mp4"}
    rows.extend([val,test])
    monkeypatch.setattr(m27,"read_manifest",lambda _:rows)
    monkeypatch.setattr(m27,"load_clip_table",lambda _:pd.DataFrame())
    monkeypatch.setattr(m27,"_read_table",lambda _:upper)
    monkeypatch.setattr(m27,"frozen_train_manifest",
                        lambda *_args:(rows[:-2],set(upper.subject_id)))
    class DummyPose:
        def __enter__(self): return self
        def __exit__(self,*_): return False
    monkeypatch.setitem(sys.modules,"mediapipe",SimpleNamespace(
        solutions=SimpleNamespace(pose=SimpleNamespace(
            Pose=lambda **_opts:DummyPose()
        ))
    ))
    monkeypatch.setitem(sys.modules,"cv2",SimpleNamespace())
    opened=[]
    def stub_read(_zip,member,*_args,**_kwargs):
        opened.append(member.filename)
        return np.ones(16),None
    monkeypatch.setattr(m27,"_read_video",stub_read)
    archive=tmp_path/"local.zip"
    with zipfile.ZipFile(archive,"w") as zipped:
        for row in rows:
            zipped.writestr(row["archive_member"],b"synthetic")
    destination=tmp_path/"features.csv"
    r=m27.extract(archive,tmp_path/"manifest",tmp_path/"original",
                  tmp_path/"upper",destination)
    assert len(opened)==len(upper)
    assert "excluded_VAL.mp4" not in opened
    assert "excluded_TEST.mp4" not in opened
    assert r["val_or_test_source_video_decoded"] is False
    assert len(pd.read_csv(destination))==len(upper)
    with pytest.raises(FileExistsError):
        m27.extract(archive,tmp_path/"manifest",tmp_path/"original",
                    tmp_path/"upper",destination)
