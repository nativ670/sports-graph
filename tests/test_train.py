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
