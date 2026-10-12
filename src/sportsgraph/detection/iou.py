"""Box geometry: IoU and format conversions.

A "box" here is always (x1, y1, x2, y2) in absolute pixels, top-left and
bottom-right corners. YOLO format stores normalized (x_center, y_center,
width, height) instead, so conversion happens at the loading boundary
(yolo_labels.py), not scattered through the evaluation logic.
"""


def yolo_to_xyxy(
    x_center: float, y_center: float, width: float, height: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Convert normalized YOLO-format box to absolute pixel corners."""
    cx, cy, w, h = x_center * img_w, y_center * img_h, width * img_w, height * img_h
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union
