"""Unit tests for training manifest preparation."""

from pathlib import Path

import yaml
from PIL import Image

from sportsgraph.detection.train import prepare_training_manifest


def test_prepare_training_manifest(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("ball\ngoalkeeper\nplayer\nreferee\n")

    # Create 10 dummy images
    for i in range(10):
        img_path = images_dir / f"frame_{i:04d}.jpg"
        Image.new("RGB", (64, 64)).save(img_path)
        (labels_dir / f"frame_{i:04d}.txt").write_text("2 0.5 0.5 0.2 0.2\n")

    manifest_path = prepare_training_manifest(tmp_path, held_out_frac=0.2)

    assert manifest_path.exists()
    assert (tmp_path / "train.txt").exists()
    assert (tmp_path / "val.txt").exists()

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    assert manifest["nc"] == 4
    assert manifest["names"] == ["ball", "goalkeeper", "player", "referee"]

    train_lines = (tmp_path / "train.txt").read_text().strip().splitlines()
    val_lines = (tmp_path / "val.txt").read_text().strip().splitlines()

    assert len(train_lines) == 8
    assert len(val_lines) == 2
    assert "frame_0008.jpg" in val_lines[0]
    assert "frame_0009.jpg" in val_lines[1]


def test_prepare_training_manifest_roboflow_splits(tmp_path: Path):
    # Setup Roboflow native train and valid folders
    train_dir = tmp_path / "train" / "images"
    valid_dir = tmp_path / "valid" / "images"
    test_dir = tmp_path / "test" / "images"
    train_dir.mkdir(parents=True)
    valid_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("ball\ngoalkeeper\nplayer\nreferee\n")

    # 3 train images, 2 valid images, 1 test image
    for i in range(3):
        Image.new("RGB", (64, 64)).save(train_dir / f"train_{i}.jpg")
    for i in range(2):
        Image.new("RGB", (64, 64)).save(valid_dir / f"val_{i}.jpg")
    Image.new("RGB", (64, 64)).save(test_dir / "test_0.jpg")

    manifest_path = prepare_training_manifest(tmp_path)

    assert manifest_path.exists()
    assert (tmp_path / "train.txt").exists()
    assert (tmp_path / "val.txt").exists()
    assert (tmp_path / "test.txt").exists()

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    assert manifest["train"] == "train.txt"
    assert manifest["val"] == "val.txt"
    assert manifest["test"] == "test.txt"

    train_lines = (tmp_path / "train.txt").read_text().strip().splitlines()
    val_lines = (tmp_path / "val.txt").read_text().strip().splitlines()
    test_lines = (tmp_path / "test.txt").read_text().strip().splitlines()

    assert len(train_lines) == 3
    assert len(val_lines) == 2
    assert len(test_lines) == 1
    assert "train_0.jpg" in train_lines[0]
    assert "val_0.jpg" in val_lines[0]
    assert "test_0.jpg" in test_lines[0]
