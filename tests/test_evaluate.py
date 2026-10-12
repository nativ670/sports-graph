import pytest

from sportsgraph.detection.evaluate import (
    GroundTruth,
    Prediction,
    compute_ap,
    compute_map,
    match_image,
)


def test_compute_ap_matches_hand_calculation():
    """A specific, hand-worked example (see the derivation alongside this code):
    5 ground truths, 8 predictions in confidence order with outcomes
    TP,TP,FP,TP,FP,FP,TP,FP. 101-point interpolated AP works out to
    472/707 ≈ 0.667610, confirmed by direct calculation independent of
    this implementation.
    """
    outcomes = [True, True, False, True, False, False, True, False]
    confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    matches = list(zip(confidences, outcomes, strict=True))

    ap = compute_ap(matches, num_gt=5)
    assert ap == pytest.approx(472 / 707, abs=1e-9)


def test_perfect_detector_has_ap_one():
    matches = [(0.9, True), (0.8, True), (0.7, True)]
    assert compute_ap(matches, num_gt=3) == pytest.approx(1.0)


def test_no_predictions_has_ap_zero():
    assert compute_ap([], num_gt=5) == pytest.approx(0.0)


def test_no_ground_truth_is_undefined():
    assert compute_ap([(0.9, False)], num_gt=0) is None


def test_match_image_greedy_assignment():
    # Two overlapping boxes, two predictions both claiming the same region
    # with different confidence: the higher-confidence one should win the
    # match, the lower-confidence duplicate should be a false positive.
    gt = [GroundTruth(box=(0, 0, 10, 10), class_id=0)]
    preds = [
        Prediction(box=(0.5, 0.5, 10.5, 10.5), confidence=0.9, class_id=0),
        Prediction(box=(0.3, 0.3, 10.3, 10.3), confidence=0.6, class_id=0),  # duplicate
    ]
    results = match_image(preds, gt, iou_threshold=0.5)
    results_by_conf = dict(results)
    assert results_by_conf[0.9] is True
    assert results_by_conf[0.6] is False  # the real GT was already claimed


def test_match_image_below_threshold_is_false_positive():
    gt = [GroundTruth(box=(0, 0, 10, 10), class_id=0)]
    preds = [Prediction(box=(100, 100, 110, 110), confidence=0.9, class_id=0)]
    results = match_image(preds, gt, iou_threshold=0.5)
    assert results == [(0.9, False)]


def test_unmatched_ground_truth_does_not_appear_in_results():
    # Two real objects, only one gets a prediction at all -> match_image
    # returns exactly one result; the second GT's absence is what makes it
    # a false negative once compute_ap divides by the full num_gt.
    gt = [
        GroundTruth(box=(0, 0, 10, 10), class_id=0),
        GroundTruth(box=(50, 50, 60, 60), class_id=0),
    ]
    preds = [Prediction(box=(0, 0, 10, 10), confidence=0.9, class_id=0)]
    results = match_image(preds, gt, iou_threshold=0.5)
    assert len(results) == 1
    assert results[0] == (0.9, True)


def test_compute_map_averages_only_classes_with_ground_truth():
    matches_by_class = {
        "player": [(0.9, True), (0.8, True)],
        "ball": [(0.9, False)],
    }
    num_gt_by_class = {"player": 2, "ball": 0}  # no real balls in this toy set
    result = compute_map(matches_by_class, num_gt_by_class)
    assert "ball" not in result  # undefined AP excluded, not silently zeroed
    assert result["player"] == pytest.approx(1.0)
    assert result["mAP"] == pytest.approx(1.0)
