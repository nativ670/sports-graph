import pytest

from sportsgraph.detection.iou import iou, yolo_to_xyxy


def test_identical_boxes_have_iou_one():
    box = (10.0, 10.0, 50.0, 50.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_non_overlapping_boxes_have_iou_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_known_partial_overlap():
    # Two 10x10 squares, offset by 5 in both axes -> 5x5 intersection = 25,
    # union = 100 + 100 - 25 = 175, IoU = 25/175 = 1/7.
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (5.0, 5.0, 15.0, 15.0)
    assert iou(box1, box2) == pytest.approx(1 / 7)


def test_touching_but_not_overlapping_is_zero():
    # Boxes share an edge (x=10) but have zero-area intersection.
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_yolo_to_xyxy_conversion():
    # Box centered at (0.5, 0.5) normalized, width=0.2, height=0.4, in a 100x200 image.
    result = yolo_to_xyxy(0.5, 0.5, 0.2, 0.4, img_w=100, img_h=200)
    assert result == pytest.approx((40.0, 60.0, 60.0, 140.0))
