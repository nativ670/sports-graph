"""Match predicted boxes to ground truth and compute AP / mAP.

This mirrors COCO's own evaluation procedure: greedy confidence-ranked
matching per image, then 101-point recall-interpolated average precision,
exactly what's described in the mAP topic. Kept separate from any
particular detector or label format, this module only knows about boxes,
confidences, and class ids.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2), absolute pixels
    confidence: float
    class_id: int


@dataclass(frozen=True)
class GroundTruth:
    box: tuple[float, float, float, float]
    class_id: int


def match_image(
    predictions: list[Prediction],
    ground_truths: list[GroundTruth],
    iou_threshold: float = 0.5,
) -> list[tuple[float, bool]]:
    """Match one image's predictions (one class) against its ground truth boxes.

    Greedy, confidence-ranked: the highest-confidence prediction gets first
    claim on the best-overlapping unmatched ground truth. A ground truth can
    only be matched once, so a second prediction on the same real object is
    a false positive (a duplicate detection), not a second true positive.

    Returns one (confidence, is_true_positive) pair per prediction. Ground
    truths left unmatched are implicitly false negatives: they contribute to
    num_gt in compute_ap but never appear in this return value.
    """
    from sportsgraph.detection.iou import iou

    ordered = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    matched_gt_indices: set[int] = set()
    results = []

    for pred in ordered:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(ground_truths):
            if idx in matched_gt_indices:
                continue
            candidate_iou = iou(pred.box, gt.box)
            if candidate_iou > best_iou:
                best_iou = candidate_iou
                best_idx = idx

        if best_idx != -1 and best_iou >= iou_threshold:
            matched_gt_indices.add(best_idx)
            results.append((pred.confidence, True))
        else:
            results.append((pred.confidence, False))

    return results


def compute_ap(matches: list[tuple[float, bool]], num_gt: int) -> float | None:
    """101-point interpolated average precision, COCO-style.

    `matches` is the concatenation of match_image()'s output across every
    image in the evaluation set, for one class. `num_gt` is the total count
    of real ground-truth boxes of that class across the same set (including
    ones no prediction ever matched).
    """
    if num_gt == 0:
        return None  # undefined: nothing of this class actually exists to find

    ordered = sorted(matches, key=lambda m: m[0], reverse=True)

    cum_tp = 0
    cum_fp = 0
    precisions = []
    recalls = []
    for _confidence, is_tp in ordered:
        if is_tp:
            cum_tp += 1
        else:
            cum_fp += 1
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / num_gt)

    recall_thresholds = [k / 100 for k in range(101)]
    interpolated = []
    for r in recall_thresholds:
        achievable = [p for p, rec in zip(precisions, recalls, strict=True) if rec >= r]
        interpolated.append(max(achievable) if achievable else 0.0)

    return sum(interpolated) / len(interpolated)


def compute_map(
    matches_by_class: dict[str, list[tuple[float, bool]]],
    num_gt_by_class: dict[str, int],
) -> dict[str, float]:
    """Per-class AP plus the mean (mAP) over classes that have any ground truth."""
    ap_by_class = {}
    for class_name, num_gt in num_gt_by_class.items():
        matches = matches_by_class.get(class_name, [])
        ap = compute_ap(matches, num_gt)
        if ap is not None:
            ap_by_class[class_name] = ap

    result = dict(ap_by_class)
    result["mAP"] = sum(ap_by_class.values()) / len(ap_by_class) if ap_by_class else 0.0
    return result
