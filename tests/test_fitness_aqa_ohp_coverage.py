import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_fitness_aqa_ohp_coverage as qa


def pose_landmarks(low=()):
    return [
        SimpleNamespace(x=.02 * i, y=.03 * i, z=.001 * i,
                        visibility=(.4 if i in low else .99))
        for i in range(33)
    ]


def test_ohp_diagnostic_baseline_exact_and_task_specific_masks():
    good = qa.classify(pose_landmarks())
    assert good[0] == "accepted"
    assert all(good[2].values())
    knee = qa.classify(pose_landmarks(low=(25,)))
    assert knee[0] == "critical_joint_low_visibility"
    assert knee[1] == ("left_knee",)
    assert knee[2]["all12_050_production"] is False
    assert knee[2]["upper8_050_elbow_candidate"] is True
    assert knee[2]["lower8_050_knee_candidate"] is False
    assert knee[2]["all12_035"] is True
    wrist = qa.classify(pose_landmarks(low=(15,)))
    assert wrist[2]["lower8_050_knee_candidate"] is True
    assert wrist[2]["upper8_050_elbow_candidate"] is False
    assert wrist[2]["all12_035"] is True
    assert qa.classify(None)[0] == "no_pose_detected"
    assert qa.classify(pose_landmarks()[:20])[0] == "invalid_landmarks_or_geometry"
    assert qa.classify(pose_landmarks(low=(11,)))[2]["all12_050_production"] is False


def test_strata_deterministic_and_cover_rare_error_modes():
    ids = [f"{i}_1" for i in range(40)]
    elbow = {k: ([[.2,.6]] if i%4 in (0,1) else []) for i,k in enumerate(ids)}
    knee = {k: ([[.2,.6]] if i%4 in (0,2) else []) for i,k in enumerate(ids)}
    a = qa.sample_clips(ids, elbow, knee, count=12, seed=42)
    b = qa.sample_clips(ids, elbow, knee, count=12, seed=42)
    assert a==b
    assert len({i for i,_ in a})==12
    assert len(set(name for _,name in a))==4
    assert all(sum(stratum==name for _,name in a)>=3 for name in {
        "both_errors_annotated", "elbows_annotated_only",
        "knees_annotated_only", "neither_named_error_annotated",
    })


def test_policy_summary_positives_recover_without_claiming_model_quality():
    good = qa.classify(pose_landmarks())
    low_knee = qa.classify(pose_landmarks(low=(25,)))
    no_pose = qa.classify(None)
    rows = [
        {"video_id":"a","stratum":"both_errors_annotated",
         "reason":good[0],"low_visibility_joints":good[1],
         "gate_flags":good[2],"error_elbows":1,"error_knees":1},
        {"video_id":"a","stratum":"both_errors_annotated",
         "reason":low_knee[0],"low_visibility_joints":low_knee[1],
         "gate_flags":low_knee[2],"error_elbows":1,"error_knees":0},
        {"video_id":"b","stratum":"neither_named_error_annotated",
         "reason":no_pose[0],"low_visibility_joints":no_pose[1],
         "gate_flags":no_pose[2],"error_elbows":0,"error_knees":0},
    ]
    summary = qa.summarize(rows, [("a","both_errors_annotated"),
                                  ("b","neither_named_error_annotated")],
                           boundary_dropped=0,failed=[])
    assert summary["reason_counts"]["no_pose_detected"]==1
    assert summary["low_visibility_joint_occurrences"]["left_knee"]==1
    assert summary["candidate_policies"]["all12_050_production"]["accepted_frames"]==1
    upper = summary["candidate_policies"]["upper8_050_elbow_candidate"]
    assert upper["accepted_frames"]==2
    assert upper["targets"]["error_elbows"]["new_positive_vs_production"]==1
    assert upper["targets"]["error_knees"]["new_positive_vs_production"]==0
    assert upper["accepted_fraction_of_eligible"] == 2/3


def make_source(tmp_path):
    base=tmp_path/"ohp"/"labeled"
    ann=base/"annotations"
    ann.mkdir(parents=True)
    train=["101_1","102_1","103_1","104_1"]
    val=["201_1","202_1","203_1","204_1"]
    test=["999_1"]
    for name,keys in (("train",train),("val",val),("test",test)):
        (ann/f"{name}_keys.json").write_text(json.dumps(keys))
    elbow={k: ([[.3,.8]] if k[:1] in ("1","2") and
               k[1:3] in ("01","03") else []) for k in train+val+test}
    knee={k: ([[.3,.8]] if k[1:3] in ("01","02") else []) for k in train+val+test}
    (ann/"error_elbows.json").write_text(json.dumps(elbow))
    (ann/"error_knees.json").write_text(json.dumps(knee))
    with zipfile.ZipFile(base/"videos.zip","w") as z:
        for k in train+val:
            z.writestr(f"videos/{k}.mp4",b"fake")
        z.writestr("videos/999_1.mp4",b"test is never decoded")
    return base


def test_end_to_end_reads_source_train_val_only(tmp_path):
    base=make_source(tmp_path)
    class Reader:
        def __init__(self):
            self.index=0
        def isOpened(self): return True
        def get(self,_): return 10.
        def read(self):
            if self.index>=6:
                return False,None
            self.index+=1
            return True,np.zeros((3,3,3),dtype=np.uint8)
        def release(self): pass
    class CV2:
        CAP_PROP_FPS=5
        COLOR_BGR2RGB=6
        @staticmethod
        def VideoCapture(path):
            assert path.endswith(".mp4")
            return Reader()
        @staticmethod
        def cvtColor(frame,_): return frame
    class Estimator:
        def __init__(self,**kwargs):
            assert kwargs["static_image_mode"] is True
            self.counter=0
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def process(self,frame):
            self.counter+=1
            if self.counter % 4 == 1:
                return SimpleNamespace(pose_landmarks=None)
            if self.counter % 4 == 2:
                points=pose_landmarks(low=(25,))
            elif self.counter % 4 == 3:
                points=pose_landmarks(low=(15,))
            else:
                points=pose_landmarks()
            return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=points))
    mp=SimpleNamespace(solutions=SimpleNamespace(
        pose=SimpleNamespace(Pose=Estimator)))
    out=tmp_path/"reports"/"ohp_quality.json"
    result=qa.audit(base/"videos.zip",base/"annotations",out,
                    clips_per_split=4,sample_fps=2.,boundary_margin=0.,
                    cv2_api=CV2,mp_api=mp)
    assert result["test_video_clips_accessed"]==0
    assert out.is_file()
    assert set(result["splits"])=={"train","val"}
    assert result["splits"]["train"]["selected_clips"]==4
    assert result["splits"]["val"]["selected_clips"]==4
    assert result["splits"]["train"]["eligible_frames"]==8
    for split in ("train","val"):
        stats=result["splits"][split]
        assert stats["failed_clip_count"]==0
        baseline=stats["candidate_policies"]["all12_050_production"]["accepted_frames"]
        assert stats["candidate_policies"]["all12_035"]["accepted_frames"]>=baseline
    assert not (out.parent/"quality_test.csv").exists()


def test_reject_invalid_sampling_config_without_opening_media(tmp_path):
    with pytest.raises(ValueError):
        qa.audit(tmp_path,tmp_path,tmp_path,sample_fps=0)
    with pytest.raises(ValueError):
        qa.audit(tmp_path,tmp_path,tmp_path,boundary_margin=float("nan"))
