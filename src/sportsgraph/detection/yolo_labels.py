"""Load ground-truth labels in YOLO 1.1 format (what CVAT exports).

The exact folder layout of a CVAT export has varied across versions and
isn't something to assume blind, confirm it against the real unzipped
export once you have it (same spirit as checking Metrica's CSV headers
before writing the adapter). This module only assumes the two things that
have stayed stable: a plain-text classes file (one class name per line,
line number = class id) and per-image label files where each line is
`class_id x_center y_center width height`, all normalized 0-1.
"""

from pathlib import Path

from PIL import Image

from sportsgraph.detection.evaluate import GroundTruth
from sportsgraph.detection.iou import yolo_to_xyxy


def load_class_names(classes_path: Path) -> dict[int, str]:
    """Parse a classes file: one name per line, 0-indexed by line number."""
    lines = Path(classes_path).read_text().splitlines()
    return {i: name.strip() for i, name in enumerate(lines) if name.strip()}


def load_yolo_label_file(
    label_path: Path, class_names: dict[int, str], img_w: int, img_h: int
) -> list[GroundTruth]:
    """Parse one YOLO-format .txt label file into absolute-pixel GroundTruth boxes."""
    path = Path(label_path)
    if not path.exists():
        return []  # a frame with no objects has no label file at all, not an error

    ground_truths = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        class_id_str, x_c, y_c, w, h = line.split()
        class_id = int(class_id_str)
        box = yolo_to_xyxy(float(x_c), float(y_c), float(w), float(h), img_w, img_h)
        ground_truths.append(GroundTruth(box=box, class_id=class_id))
    return ground_truths


def image_size(image_path: Path) -> tuple[int, int]:
    """Return (width, height) of an image file, needed to de-normalize YOLO boxes."""
    with Image.open(image_path) as img:
        return img.size
