import pandas as pd
import pytest

from gym_pose.dataset import manifest_template, read_manifest, summarize_manifest


def sample_manifest(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    samples = []
    for i in range(4):
        path = images / f"image{i}.jpg"
        path.write_bytes(f"example-image-{i}".encode())
        samples.append({"sample_id": f"im{i}", "relative_path": path.name,
                        "exercise": "squat", "label": "correct" if i % 2 else "incorrect",
                        "subject_id": f"s{i // 2}", "session_id": f"session{i // 2}",
                        "camera_view": "side"})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(samples).to_csv(manifest, index=False)
    return images, manifest


def test_template(tmp_path):
    path = manifest_template(tmp_path / "new.csv")
    assert "subject_id" in pd.read_csv(path).columns


def test_manifest_counts_and_hashes(tmp_path):
    images, path = sample_manifest(tmp_path)
    valid = read_manifest(path, images)
    assert len(valid) == 4
    assert valid.image_sha256.str.len().eq(64).all()
    assert summarize_manifest(path, images)["subjects"] == 2


def test_duplicate_image_rejected(tmp_path):
    images, path = sample_manifest(tmp_path)
    (images / "image1.jpg").write_bytes((images / "image0.jpg").read_bytes())
    with pytest.raises(ValueError, match="Duplicate image bytes"):
        read_manifest(path, images)


def test_path_traversal_rejected(tmp_path):
    images, path = sample_manifest(tmp_path)
    df = pd.read_csv(path)
    df.loc[0, "relative_path"] = "../escape.jpg"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_manifest(path, images)


def test_session_conflict_rejected(tmp_path):
    images, path = sample_manifest(tmp_path)
    df = pd.read_csv(path)
    df.loc[0, "session_id"] = "session1"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="session_id"):
        read_manifest(path, images)
