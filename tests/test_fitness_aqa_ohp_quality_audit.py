"""Synthetic OHP quality-gate investigation: no real media, labels or test access."""
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from gym_pose.ohp_quality import assess_ohp_pose_quality
from scripts.audit_fitness_aqa_ohp_quality import (
    STRATA, audit, select_videos, summarize,
)


def landmarks(*, low_joint=None, level=.4):
    points = [SimpleNamespace(x=.02 * i,y=.03 * i,z=.01 * i,visibility=1.)
              for i in range(33)]
    if low_joint is not None:
        points[low_joint].visibility=level
    return points


def test_ohp_gate_provenance_and_first_reason_priority():
    assert assess_ohp_pose_quality(None)["reason"] == "no_pose_detected"
    assert assess_ohp_pose_quality(landmarks())["reason"] == "accepted"
    knee = assess_ohp_pose_quality(landmarks(low_joint=25))
    assert knee["reason"] == "squat_base_joint_low_visibility"
    assert "left_knee" in knee["low_visibility_joints"]
    wrist = assess_ohp_pose_quality(landmarks(low_joint=15))
    assert wrist["reason"] == "ohp_upper_joint_low_visibility"
    assert "left_wrist" in wrist["low_visibility_joints"]
    assert assess_ohp_pose_quality(landmarks(low_joint=15),
                                   min_visibility=.35)["reason"] == "accepted"
    mixed=landmarks(low_joint=25)
    mixed[15].visibility=.3
    both=assess_ohp_pose_quality(mixed)
    assert both["reason"] == "squat_base_joint_low_visibility"
    assert "left_wrist" in both["low_visibility_joints"]
    assert assess_ohp_pose_quality(landmarks()[:32])["reason"] == "invalid_landmark_structure"
    invalid=landmarks()
    invalid[11].x=float("nan")
    assert assess_ohp_pose_quality(invalid)["reason"] == "nonfinite_coordinates_or_visibility"


def test_annotation_strata_are_deterministic_and_balanced():
    ids=[f"{i}_1" for i in range(30)]
    elbows={key:([[0.,.5]] if i%3==1 else []) for i,key in enumerate(ids)}
    knees={key:([[0.,.5]] if i%3==0 else []) for i,key in enumerate(ids)}
    selected=select_videos(ids,elbows,knees,count=9,seed=42)
    assert selected == select_videos(ids,elbows,knees,count=9,seed=42)
    assert len(selected)==9
    assert {key for key,_ in selected} <= set(ids)
    assert sorted([s for _,s in selected]).count("knees_annotated_video")==3
    assert sorted([s for _,s in selected]).count("elbows_only_annotated_video")==3
    assert sorted([s for _,s in selected]).count("neither_error_annotated_video")==3


def test_zero_accepted_errors_do_not_hide_positive_eligible_frames():
    rows=[
        {"video_id":"a","rejection_reason":"accepted","error_elbows":0,
         "error_knees":0,"accepted_at_baseline":True,
         "accepted_at_diagnostic_cutoff":True,"low_visibility_joints":[],
         "video_annotation_stratum":STRATA[2]},
        {"video_id":"b","rejection_reason":"squat_base_joint_low_visibility",
         "error_elbows":0,"error_knees":1,"accepted_at_baseline":False,
         "accepted_at_diagnostic_cutoff":True,
         "low_visibility_joints":["left_knee"],
         "video_annotation_stratum":STRATA[0]},
    ]
    r=summarize(rows,[("a",STRATA[2]),("b",STRATA[0])],
                rejected_boundary=0,failures=[],diagnostic_visibility=.35)
    assert r["accepted_baseline_frames"]==1
    knee=r["by_error_annotation_at_timestamp"]["error_knees"]["1"]
    assert knee["eligible_frames"]==1
    assert knee["accepted_baseline"]==0
    assert knee["accepted_alternative"]==1


def make_sources(tmp_path):
    base=tmp_path/"ohp"/"labeled"
    annotations=base/"annotations"
    annotations.mkdir(parents=True)
    ids={
      "train":["101_1","102_1","103_1"],
      "val":["201_1","202_1","203_1"],
      "test":["999_1"],
    }
    for split,items in ids.items():
        (annotations/f"{split}_keys.json").write_text(json.dumps(items))
    elbow={key:([[.25,.8]] if key in ("102_1","202_1") else [])
           for split in ids for key in ids[split]}
    knee={key:([[.25,.8]] if key in ("101_1","201_1") else [])
          for split in ids for key in ids[split]}
    (annotations/"error_elbows.json").write_text(json.dumps(elbow))
    (annotations/"error_knees.json").write_text(json.dumps(knee))
    with zipfile.ZipFile(base/"videos.zip","w") as z:
        for key in [*ids["train"],*ids["val"],*ids["test"],"9999_1"]:
            z.writestr(f"videos/{key}.mp4",b"fixture")
    return base/"videos.zip",annotations


class Reader:
    def __init__(self): self.pos=0
    def isOpened(self): return True
    def get(self, prop): return 10.
    def read(self):
        if self.pos >= 6: return False,None
        self.pos+=1
        return True,np.zeros((2,2,3),dtype=np.uint8)
    def release(self): pass

class CV2:
    CAP_PROP_FPS=5
    COLOR_BGR2RGB=6
    @staticmethod
    def VideoCapture(_): return Reader()
    @staticmethod
    def cvtColor(image,_): return image

class Pose:
    def __init__(self,**kwargs):
        assert kwargs.get("static_image_mode") is True
        self.i=0
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def process(self, _):
        self.i+=1
        # Repeat: no pose, bad ankle, bad wrist, accepted
        mode=self.i%4
        if mode==1: return SimpleNamespace(pose_landmarks=None)
        lm=landmarks(low_joint=27 if mode==2 else 15 if mode==3 else None)
        return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=lm))

MP=SimpleNamespace(solutions=SimpleNamespace(pose=SimpleNamespace(Pose=Pose)))


def test_realistic_pilot_without_decoding_test(tmp_path):
    archive,annotations=make_sources(tmp_path)
    out=tmp_path/"report"/"quality.json"
    result=audit(archive,annotations,out,videos_per_split=3,
                 seed=42,boundary_margin=0,
                 cv2_api=CV2,mp_api=MP)
    assert result["source_test_clips_decoded"]==0
    assert set(result["splits"])=={"train","val"}
    assert out.is_file()
    for split in ("train","val"):
        row=result["splits"][split]
        assert row["selected_videos"]==3
        assert row["videos_failed"]==0
        assert row["eligible_frames"]==6
        assert row["alternative_coverage"]>=row["baseline_coverage"]
        assert sum(row["rejection_reason_counts"].values())==row["eligible_frames"]
        assert row["by_publisher_video_annotation_stratum"][STRATA[0]]["selected_video_count"]==1
        assert row["by_publisher_video_annotation_stratum"][STRATA[1]]["selected_video_count"]==1
        assert row["by_publisher_video_annotation_stratum"][STRATA[2]]["selected_video_count"]==1
    assert not (tmp_path/"report"/"test.json").exists()


def test_test_split_not_exposed_and_bad_parameters_fail(tmp_path):
    with pytest.raises(ValueError):
        audit(tmp_path,tmp_path,tmp_path,videos_per_split=0)
    with pytest.raises(ValueError):
        audit(tmp_path,tmp_path,tmp_path,diagnostic_visibility=.8)
