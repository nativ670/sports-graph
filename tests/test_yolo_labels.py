from pathlib import Path

from PIL import Image

from sportsgraph.detection.yolo_labels import (
    image_size,
    load_class_names,
    load_yolo_label_file,
)


def test_load_class_names(tmp_path: Path):
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("player\nball\n")
    result = load_class_names(classes_file)
    assert result == {0: "player", 1: "ball"}


def test_load_yolo_label_file(tmp_path: Path):
    label_file = tmp_path / "frame_0001.txt"
    # One player (class 0) centered at (0.5, 0.5), one ball (class 1) near the corner.
    label_file.write_text("0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.02 0.02\n")
    class_names = {0: "player", 1: "ball"}

    result = load_yolo_label_file(label_file, class_names, img_w=100, img_h=200)

    assert len(result) == 2
    assert result[0].class_id == 0
    assert result[0].box == (40.0, 60.0, 60.0, 140.0)  # matches the hand-checked iou.py conversion
    assert result[1].class_id == 1


def test_load_yolo_label_file_missing_file_returns_empty(tmp_path: Path):
    # A frame with zero objects legitimately has no label file at all.
    result = load_yolo_label_file(tmp_path / "does_not_exist.txt", {0: "player"}, 100, 100)
    assert result == []


def test_image_size(tmp_path: Path):
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (640, 480)).save(img_path)
    assert image_size(img_path) == (640, 480)
