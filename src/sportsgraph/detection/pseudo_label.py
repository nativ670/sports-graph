"""Generate draft YOLO-format labels from a pretrained model, for import into
CVAT as a starting point to correct rather than draw from scratch.

Split the same way as run_baseline_eval.py: the pure formatting/filtering
logic here is fully unit-tested against synthetic detections. The one part
that can't be tested without a real model and real images is `main()`'s
call into ultralytics, flagged clearly below.
"""

import argparse
from pathlib import Path

from sportsgraph.detection.yolo_labels import load_class_names

COCO_TO_OUR_CLASS = {"person": "player", "sports ball": "ball"}


def format_yolo_line(
    class_id: int, x_center: float, y_center: float, width: float, height: float
) -> str:
    """One line of a YOLO-format label file: `class_id x_center y_center width height`."""
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def pseudo_label_lines(
    detections: list[tuple[str, tuple[float, float, float, float], float]],
    our_class_name_to_id: dict[str, int],
    conf_threshold: float = 0.25,
) -> list[str]:
    """Turn a list of (coco_class_name, (x_c, y_c, w, h) normalized, confidence)
    detections into formatted YOLO label lines, keeping only the classes we
    care about (player/ball) above the confidence threshold.

    A moderate threshold (0.25, ultralytics' own default) is deliberate here:
    this is meant to be "mostly right, correct the mistakes," not the
    near-zero threshold run_baseline_eval.py uses to sweep the full
    precision-recall curve for AP.
    """
    lines = []
    for coco_name, (x_c, y_c, w, h), conf in detections:
        our_class = COCO_TO_OUR_CLASS.get(coco_name)
        if our_class is None or our_class not in our_class_name_to_id or conf < conf_threshold:
            continue
        class_id = our_class_name_to_id[our_class]
        lines.append(format_yolo_line(class_id, x_c, y_c, w, h))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-labels-dir", type=Path, required=True)
    parser.add_argument("--classes-file", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    # --- Everything below this line is the untested glue: a real model call. ---
    from ultralytics import YOLO

    class_names = load_class_names(args.classes_file)
    our_class_name_to_id = {name: cid for cid, name in class_names.items()}
    args.output_labels_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    image_paths = sorted(args.images_dir.glob("*.jpg")) + sorted(args.images_dir.glob("*.png"))

    for image_path in image_paths:
        result = model.predict(str(image_path), conf=0.05, verbose=False)[0]
        detections = [
            (model.names[int(box.cls[0])], tuple(box.xywhn[0].tolist()), float(box.conf[0]))
            for box in result.boxes
        ]
        lines = pseudo_label_lines(detections, our_class_name_to_id, args.conf)

        out_path = args.output_labels_dir / f"{image_path.stem}.txt"
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"{image_path.name}: {len(lines)} draft boxes -> {out_path.name}")


if __name__ == "__main__":
    main()
