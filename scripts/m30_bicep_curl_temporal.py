"""M30 Bicep Curl - TRAIN-only elbow temporal feature ablation.

Uses publisher Gym7 bicep_curl clips and frozen original M20 seed42 TRAIN
subjects only. The original M20 validation/TEST videos and predictions are
not decoded, fitted, accessed, or scored.

The original 264 clip features remain unchanged. This pilot extracts 16
time-order and tracking descriptors of elbow motion then compares 264 vs
276 kinematics vs 268 tracking proxies vs 280 all on same TRAIN clip IDs
with 5-fold subject GroupKFold and LOSO. Research-only, no web good/bad AI.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from gym_pose.gym7_research import read_manifest, VIDEO_TYPES, frame_pose_vector, FEATURES
from gym_pose.gym7_training import load_clip_table, split_by_subject
from gym_pose.gym7_m24_cv import _metrics
from scripts.m27_pushup_temporal_train import MOTION_COLS, motion_summary

SCHEMA = "gym7-bicep-curl-temporal16-v1"
SEED = 42
TREES = 300
META = ("video_id", "subject_id", "exercise", "label", "view")
MOTION_FILE_COLS = (*META, *MOTION_COLS)
UPPER_SIDES = ((11,13,15,23), (12,14,16,24))
FAMILIES = {
    "baseline_264": FEATURES,
    "kinematic_276": (*FEATURES, *MOTION_COLS[:12]),
    "tracking_proxy_268": (*FEATURES, *MOTION_COLS[12:]),
    "temporal_280": (*FEATURES, *MOTION_COLS),
}


def frozen_curl_train(manifest, original):
    parts = split_by_subject(original, seed=SEED)
    identities = {mode: set(frame.subject_id) for mode,frame in parts.items()}
    if (identities["train"] & identities["val"] or
        identities["train"] & identities["test"] or
        identities["val"] & identities["test"]):
        raise ValueError("Original M20 TRAIN/VAL/TEST subject leakage")
    by_vid={r["video_id"]:r for r in manifest}
    if len(by_vid)!=len(manifest):
        raise ValueError("Duplicate source video")
    for record in original.itertuples(index=False):
        source=by_vid.get(record.video_id)
        if source is None or any(getattr(record,k)!=source[k] for k in META):
            raise ValueError("Original M20 subject, view or label mismatch")
    eligible = original.loc[
        original.exercise.eq("bicep_curl") &
        original.subject_id.isin(identities["train"])
    ]
    if eligible.empty or eligible.subject_id.nunique() < 5:
        raise ValueError("Insufficient source TRAIN Bicep Curl people")
    selected = sorted(eligible.video_id.tolist())
    return selected, eligible, by_vid


def curl_motion_frame(landmarks):
    """Only accepted source Curl frames contribute motion geometry."""
    if frame_pose_vector(landmarks, "bicep_curl") is None:
        return None
    points=np.asarray([[p.x,p.y,p.z] for p in landmarks],dtype=float)
    vis=np.asarray([p.visibility for p in landmarks],dtype=float)
    side_id=int(np.argmax([min(vis[list(side)]) for side in UPPER_SIDES]))
    shoulder, elbow, wrist, _hip = UPPER_SIDES[side_id]
    a=points[shoulder,:2]-points[elbow,:2]
    b=points[wrist,:2]-points[elbow,:2]
    denom=float(np.linalg.norm(a)*np.linalg.norm(b))
    if not math.isfinite(denom) or denom < 1e-6:
        return None
    angle=float(np.degrees(np.arccos(np.clip(np.dot(a,b)/denom,-1,1))))
    center_sh=(points[11]+points[12])/2
    center_hip=(points[23]+points[24])/2
    scale=float(np.linalg.norm(center_sh[:2]-center_hip[:2]))
    if scale < .03:
        return None
    distance=float(np.linalg.norm(points[wrist,:2]-points[shoulder,:2])/scale)
    if not np.isfinite([angle,distance]).all():
        return None
    return angle,distance,side_id


def _extract_video(zipped, member, detector, cv2, *, sample_fps=2.,
                   max_frames=240, max_video_mb=500):
    if (member is None or member.is_dir() or
        member.file_size > max_video_mb*1024*1024 or
        PurePosixPath(member.filename).suffix.lower() not in VIDEO_TYPES):
        return None,"missing_or_oversize_source_video"
    sequence=[]
    sampled=0
    with tempfile.TemporaryDirectory(prefix="m30_curl_train_") as temp:
        video=Path(temp)/("clip"+PurePosixPath(member.filename).suffix.lower())
        with zipped.open(member) as source,video.open("xb") as dst:
            shutil.copyfileobj(source,dst,length=1024*1024)
        capture=cv2.VideoCapture(str(video))
        try:
            fps=float(capture.get(cv2.CAP_PROP_FPS))
            if not capture.isOpened() or not math.isfinite(fps) or fps<=0:
                return None,"unreadable_video"
            stride=max(1,round(fps/sample_fps))
            frame_index=0
            while sampled<max_frames:
                ok,image=capture.read()
                if not ok:break
                frame_index+=1
                if (frame_index-1)%stride:continue
                sampled+=1
                result=detector.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                vector=curl_motion_frame(
                    result.pose_landmarks.landmark if result.pose_landmarks else None
                )
                if vector is not None:
                    sequence.append(((frame_index-1)/fps,*vector))
        finally:
            capture.release()
    return motion_summary(sequence,sampled),None


def extract(archive:Path,manifest_path:Path,original_path:Path,destination:Path,
            *,sample_fps=2.,max_frames=240,max_video_mb=500):
    if destination.exists():
        raise FileExistsError("M30 original extracted TRAIN data already exists")
    if not 0<sample_fps<=10 or not 4<=max_frames<=2000 or max_video_mb<5:
        raise ValueError("Invalid sampling limits")
    manifest=read_manifest(manifest_path)
    original=load_clip_table(original_path)
    ids,train,by_vid=frozen_curl_train(manifest,original)
    try:
        import cv2
        import mediapipe as mp
    except ImportError as err:
        raise RuntimeError("Activate .venv-m20 with camera dependencies") from err
    rows=[]
    failures=Counter()
    with zipfile.ZipFile(archive) as zipped:
        members={item.filename:item for item in zipped.infolist()}
        with mp.solutions.pose.Pose(
            static_image_mode=True,model_complexity=1,
            min_detection_confidence=.5
        ) as detector:
            for vid in ids:
                source=by_vid[vid]
                vector,error=_extract_video(
                    zipped,members.get(source["archive_member"]),detector,cv2,
                    sample_fps=sample_fps,max_frames=max_frames,
                    max_video_mb=max_video_mb
                )
                if error or vector is None:
                    failures[error or "insufficient_elbow_frames"]+=1
                    continue
                rows.append({**{k:source[k] for k in META},
                             **dict(zip(MOTION_COLS,map(float,vector)))})
    if len(rows)!=len(ids):
        raise ValueError(
            f"M30 motion extraction accepted {len(rows)}/{len(ids)} "
            f"frozen TRAIN Curl clips; errors={dict(failures)}. "
            "No partial CSV written; do not silently change evaluation cohort."
        )
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=MOTION_FILE_COLS)
        w.writeheader()
        w.writerows(rows)
    return {
        "schema":SCHEMA,
        "original_m20_seed":SEED,
        "original_train_subjects_for_curl":int(train.subject_id.nunique()),
        "original_train_curl_clips":len(ids),
        "motion_clips":len(rows),
        "original_VAL_or_TEST_decoded":False,
        "source_quality_labels":"publisher per-video good/bad, not specific elbow faults",
        "output":str(destination)
    }


def _read_motion(path):
    frame=pd.read_csv(path,keep_default_na=False)
    if tuple(frame.columns)!=MOTION_FILE_COLS or frame.empty or frame.video_id.duplicated().any():
        raise ValueError("Invalid M30 16-motion feature schema")
    v=frame.loc[:,MOTION_COLS].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(v).all():
        raise ValueError("Nonfinite M30 Curl motion feature values")
    return frame.set_index("video_id",drop=False)


def compare(motion_path:Path,manifest_path:Path,original_path:Path):
    manifest=read_manifest(manifest_path)
    original=load_clip_table(original_path)
    ids,train,by_vid=frozen_curl_train(manifest,original)
    motion=_read_motion(motion_path)
    if set(motion.index)!=set(ids):
        raise ValueError("M30 motion and original M20 Curl TRAIN video cohorts differ")
    train=train.set_index("video_id",drop=False).loc[ids]
    for vid in ids:
        if any(motion.at[vid,k]!=train.at[vid,k] for k in META):
            raise ValueError("M30 motion TRAIN person or label metadata mismatch")
    data=train.loc[:,(*META,*FEATURES)].join(motion.loc[ids,MOTION_COLS],how="inner")
    if len(data)!=len(ids) or not np.isfinite(
        data.loc[:,FAMILIES["temporal_280"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("M30 joined clip feature rows are missing/nonfinite")
    truth={vid:int(data.at[vid,"label"]=="bad") for vid in ids}
    result={}
    for cv_name,splitter in {
        "group5":GroupKFold(n_splits=5),
        "loso":LeaveOneGroupOut()
    }.items():
        out={name:{} for name in FAMILIES}
        folds=0
        for fit_idx,held_idx in splitter.split(data,groups=data.subject_id):
            folds+=1
            if set(data.iloc[fit_idx].subject_id)&set(data.iloc[held_idx].subject_id):
                raise AssertionError("Person included in model training and holdout")
            y=data.iloc[fit_idx].label.map({"good":0,"bad":1}).to_numpy(dtype=int)
            if set(y)!={0,1}:
                raise ValueError("A training fold lacks one quality label")
            for name,columns in FAMILIES.items():
                rf=RandomForestClassifier(
                    n_estimators=TREES,min_samples_leaf=2,class_weight="balanced",
                    random_state=SEED,n_jobs=1
                )
                rf.fit(data.iloc[fit_idx].loc[:,columns].to_numpy(dtype=float),y)
                proba=rf.predict_proba(
                    data.iloc[held_idx].loc[:,columns].to_numpy(dtype=float)
                )[:,list(rf.classes_).index(1)]
                for j,p in zip(held_idx,proba):
                    vid=ids[j]
                    if vid in out[name]:
                        raise AssertionError("Duplicate OOF Curl prediction")
                    out[name][vid]=float(p)
        if any(set(scores)!=set(ids) for scores in out.values()):
            raise AssertionError("Incomplete heldout Curl cohort")
        result[cv_name]={
            "folds":folds,
            "models":{
                name:{
                    "all_paired":_metrics([(truth[vid],score[vid]) for vid in ids]),
                    "by_view":{
                        view:_metrics([
                            (truth[vid],score[vid]) for vid in ids
                            if data.at[vid,"view"]==view
                        ]) for view in ("front","side","diagonal")
                    }
                } for name,score in out.items()
            }
        }
    return {
        "study":"M30 original TRAIN-only Bicep Curl elbow temporal analysis",
        "motion_schema":SCHEMA,
        "source":"Mendeley Gym7 v3",
        "label_unit":"video good/bad, not named form error",
        "subjects":int(data.subject_id.nunique()),
        "paired_clips":len(ids),
        "feature_sets":{k:len(v) for k,v in FAMILIES.items()},
        "models":result,
        "original_M20_VAL_or_TEST_scored":False,
        "source_video_decoded_during_compare":False,
        "public_model_deployed":False,
        "caution":"Post-VAL exploratory TRAIN-only subject-heldout performance. "
                  "Do not claim reliable Bicep Curl correctness; independent "
                  "new-subject testing and browser/source feature parity needed."
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    actions=p.add_subparsers(dest="action",required=True)
    e=actions.add_parser("extract")
    e.add_argument("--zip",type=Path,default=Path(
        "data/external/mendeley_gym7/Dataset Exercise-wise.zip"))
    e.add_argument("--manifest",type=Path,default=Path(
        "data/external/mendeley_gym7/m20_manifest.csv"))
    e.add_argument("--original",type=Path,default=Path(
        "data/features/m20_gym7/m20_clip_features.csv"))
    e.add_argument("--output",type=Path,default=Path(
        "data/features/m30_bicep_curl_train_motion_v1.csv"))
    c=actions.add_parser("compare")
    c.add_argument("--motion",type=Path,default=Path(
        "data/features/m30_bicep_curl_train_motion_v1.csv"))
    c.add_argument("--manifest",type=Path,default=Path(
        "data/external/mendeley_gym7/m20_manifest.csv"))
    c.add_argument("--original",type=Path,default=Path(
        "data/features/m20_gym7/m20_clip_features.csv"))
    c.add_argument("--report",type=Path,default=Path(
        "data/processed/m30_curl_train_cv.json"))
    opts=p.parse_args(argv)
    try:
        if opts.action=="extract":
            out=extract(opts.zip,opts.manifest,opts.original,opts.output)
        else:
            if opts.report.exists():
                raise FileExistsError("M30 CV report exists; will not overwrite")
            out=compare(opts.motion,opts.manifest,opts.original)
            opts.report.parent.mkdir(parents=True,exist_ok=True)
            opts.report.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    except (OSError,ValueError,RuntimeError,AssertionError,zipfile.BadZipFile) as err:
        print(f"M30 {opts.action} stopped: {err}",file=sys.stderr)
        return 1
    print(json.dumps(out,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
