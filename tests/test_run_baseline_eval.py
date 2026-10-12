"""Integration test for the orchestration glue, using a stub model instead of
a real ultralytics YOLO (so this runs with no network access and no GPU).
Confirms the per-class filtering and matching wiring is correct, independent
of whether the real model or real CVAT export cooperate later.
"""

from pathlib import Path

from PIL import Image

from sportsgraph.detection.run_baseline_eval import evaluate_split


class FakeTensor:
    """Mimics just enough of a torch tensor's interface for our code's needs."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values

    def __getitem__(self, i):
        return self._values[i]


class FakeBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [FakeTensor(xyxy)]
        self.conf = [conf]
        self.cls = [cls]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Stands in for ultralytics.YOLO: fixed class names, scripted predictions per image."""

    names = {0: "person", 32: "sports ball"}

    def __init__(self, predictions_by_image_name):
        self._predictions_by_image_name = predictions_by_image_name

    def predict(self, image_path, conf, verbose):
        name = Path(image_path).name
        return [FakeResult(self._predictions_by_image_name.get(name, []))]


def test_evaluate_split_wires_matching_correctly(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    # One 100x100 frame: a labeled player at (10,10)-(30,30), no ball.
    img_path = images_dir / "frame_0001.jpg"
    Image.new("RGB", (100, 100)).save(img_path)
    (labels_dir / "frame_0001.txt").write_text("0 0.2 0.2 0.2 0.2\n")  # class 0 = player

    class_names = {0: "player", 1: "ball"}

    # The fake model correctly finds the player (class 0 = COCO "person"),
    # and also hallucinates a "sports ball" that doesn't correspond to any
    # labeled ball -- exactly the kind of false positive this is meant to catch.
    fake_model = FakeModel(
        {
            "frame_0001.jpg": [
                FakeBox(xyxy=[10.0, 10.0, 30.0, 30.0], conf=0.9, cls=0),  # matches the real player
                FakeBox(xyxy=[80.0, 80.0, 90.0, 90.0], conf=0.5, cls=32),  # spurious ball
            ]
        }
    )

    result = evaluate_split(
        image_paths=[img_path],
        labels_dir=labels_dir,
        class_names=class_names,
        model=fake_model,
        coco_id_by_our_class={"player": 0, "ball": 32},
        iou_threshold=0.5,
    )

    assert result["player"] == 1.0  # perfect detection of the one real player
    assert (
        "ball" not in result
    )  # no ground-truth balls in this frame -> undefined, correctly excluded
