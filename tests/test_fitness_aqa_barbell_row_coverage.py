"""No downloaded dataset, no original source pixels, no model training."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_fitness_aqa_barbell_row_coverage as audit


def make_pose(low=(), *, corrupt=False):
    joints = [
        SimpleNamespace(
            x=.03 * i, y=.015 * (i + (i % 3)), z=.001 * i,
            visibility=.3 if i in low else .99,
        ) for i in range(33)
    ]
    if corrupt:
        joints[11].x = float("nan")
    return joints


def test_policy_agrees_with_original_gate_and_shows_missing_ankles():
    good = audit.assess_pose(make_pose())
    assert good[0] == "accepted_by_current_32_features"
    assert all(good[2].values())
    ankles = audit.assess_pose(make_pose(low=(27, 28)))
    assert ankles[0] == "critical_joint_low_visibility"
    assert ankles[2]["all8_050_production"] is False
    assert ankles[2]["all8_035"] is False
    assert ankles[2]["all8_025"] is True
    assert ankles[2]["six_no_ankles_050"] is True
    assert ankles[2]["torso4_050"] is True
    assert ankles[1] == ("left_ankle", "right_ankle")


def test_missing_knees_still_allows_only_torso4_candidate():
    knee = audit.assess_pose(make_pose(low=(25,)))
    assert not knee[2]["six_no_ankles_050"]
    assert knee[2]["torso4_050"]
    assert knee[2]["all8_025"]
    assert not knee[2]["all8_035"]
    assert audit.assess_pose(None)[0] == "no_pose_detected"
    assert audit.assess_pose(make_pose()[:30])[0] == "invalid_required_joint_geometry"
    assert audit.assess_pose(make_pose(corrupt=True))[0] == "invalid_required_joint_geometry"


def test_summary_uses_original_label_conditioned_denominators():
    samples = {
        "201_1_0": (make_pose(), 1),
        "201_1_1": (make_pose(low=(27,)), 0),
        "201_1_2": (make_pose(low=(25,)), 1),
        "201_1_3": (None, 0),
    }
    rows, labels = [], {}
    for image_id, (landmarks, label) in samples.items():
        reason, joints, gates = audit.assess_pose(landmarks)
        rows.append({
            "image_id": image_id,
            "reason": reason, "low_visibility_joints": joints,
            "gate_flags": gates,
        })
        labels[image_id] = label
    r = audit.summarize(rows, selected=set(samples), labels=labels)
    assert r["selected_positive"] == 2 and r["selected_negative"] == 2
    assert r["reasons"]["no_pose_detected"] == 1
    assert r["reasons"]["critical_joint_low_visibility"] == 2
    base = r["candidate_admission_policies"]["all8_050_production"]
    six = r["candidate_admission_policies"]["six_no_ankles_050"]
    torso = r["candidate_admission_policies"]["torso4_050"]
    assert base["accepted_images"] == 1 and base["positive_accepted"] == 1
    assert six["accepted_images"] == 2 and six["new_negative_vs_production"] == 1
    assert torso["accepted_images"] == 3
    assert torso["positive_accepted"] == 2
    assert torso["positive_acceptance_rate"] == 1
    assert r["low_visibility_joint_occurrences"]["left_ankle"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        audit.summarize(rows + rows[:1], selected=set(samples), labels=labels)


def make_dataset(tmp_path: Path, *, conflict=False):
    root = tmp_path / "barbell_row" / "labeled"
    ann = root / "annotations"
    split_keys = {
        "train": [f"201_1_{i}" for i in range(4)],
        "val": [f"301_1_{i}" for i in range(4)],
        "test": [f"901_1_{i}" for i in range(2)],
    }
    lumbar_splits = {k: list(v) for k, v in split_keys.items()}
    if conflict:
        lumbar_splits["train"][0] = "301_1_0"
        lumbar_splits["val"][0] = "201_1_0"
    all_keys = set().union(*split_keys.values())
    for task, splits in (
        ("torso_angle", split_keys), ("lumbar", lumbar_splits)
    ):
        for split, ids in splits.items():
            target = ann / "splits" / task / f"{split}_ids.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(ids))
    for task, label_file in (
        ("torso_angle", "labels_torso_angle_error.json"),
        ("lumbar", "labels_lumbar_error.json"),
    ):
        labels = {
            key: int((int(key.rsplit("_", 1)[-1]) % 2 == 0)
                     if task == "torso_angle" else
                     (int(key.rsplit("_", 1)[-1]) % 2 == 1))
            for key in all_keys
        }
        (ann / label_file).write_text(json.dumps(labels))
    with zipfile.ZipFile(root / "images.zip", "w") as archive:
        for key in sorted(all_keys):
            archive.writestr(f"images/{key}.jpg", key.encode())
    return root


class FakeCV2:
    IMREAD_COLOR = 1
    COLOR_BGR2RGB = 4
    decoded_ids = []
    @classmethod
    def imdecode(cls, data, _flag):
        image_id = bytes(data).decode()
        if image_id.startswith("901"):
            raise AssertionError("TEST IMAGE PIXELS SHOULD NEVER BE DECODED")
        cls.decoded_ids.append(image_id)
        code = int(image_id.rsplit("_", 1)[-1])
        return np.full((4, 4, 3), code, dtype=np.uint8)
    @staticmethod
    def cvtColor(img, code):
        return img


class FakePose:
    def __init__(self, **options):
        assert options["static_image_mode"] is True
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def process(self, frame):
        code = int(frame[0, 0, 0])
        points = (
            make_pose() if code == 0
            else make_pose(low=(27,)) if code == 1
            else make_pose(low=(25,)) if code == 2
            else None
        )
        return SimpleNamespace(
            pose_landmarks=(
                SimpleNamespace(landmark=points) if points else None
            )
        )


FakeMP = SimpleNamespace(
    solutions=SimpleNamespace(pose=SimpleNamespace(Pose=FakePose))
)


def test_complete_mock_audit_respects_train_val_task_labels_and_locked_test(tmp_path):
    base = make_dataset(tmp_path)
    FakeCV2.decoded_ids = []
    outfile = tmp_path / "artifacts" / "row_coverage.json"
    r = audit.audit(
        base / "images.zip", base / "annotations", outfile,
        images_per_task_split=4, seed=42,
        cv2_api=FakeCV2, mp_api=FakeMP,
    )
    assert outfile.is_file()
    assert r["original_publisher_test_images_decoded"] == 0
    assert r["separate_publisher_task_splits_preserved"]
    assert r["cross_task_exact_image_split_conflicts"] == 0
    assert len(FakeCV2.decoded_ids) == 8  # decode shared images ONCE per split
    assert not any(k.startswith("901") for k in FakeCV2.decoded_ids)
    for split in ("train", "val"):
        result = r["splits"][split]
        assert result["unique_source_images_requested"] == 4
        assert result["image_decode_or_assessment_failures"] == 0
        for task in ("torso_angle", "lumbar"):
            item = result["tasks"][task]
            assert item["selected_images"] == 4
            assert item["selected_positive"] == 2
            assert item["reasons"]["no_pose_detected"] == 1
            assert item["candidate_admission_policies"]["all8_050_production"]["accepted_images"] == 1
            assert item["candidate_admission_policies"]["torso4_050"]["accepted_images"] == 3


def test_source_cross_target_conflict_blocks_even_audit(tmp_path):
    root = make_dataset(tmp_path, conflict=True)
    with pytest.raises(ValueError, match="split conflicts"):
        audit.audit(
            root / "images.zip", root / "annotations",
            tmp_path / "out.json", cv2_api=FakeCV2, mp_api=FakeMP)


@pytest.mark.parametrize("kwargs", [
    {"images_per_task_split": 0}, {"images_per_task_split": -1},
    {"max_image_mb": 0}, {"max_pixels": -5}, {"seed": -1},
])
def test_invalid_input_fails_before_opening_images(tmp_path, kwargs):
    with pytest.raises(ValueError, match="Invalid audit"):
        audit.audit(tmp_path, tmp_path, tmp_path / "out.json", **kwargs)
