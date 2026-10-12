"""Evaluate a pretrained YOLO model against hand-labeled CVAT ground truth.

Every building block here (iou, matching, AP/mAP, the time-based split,
label loading) is already unit-tested against synthetic data. The one part
that can't be tested until the real CVAT export exists is this script's
glue: the folder layout and the actual model inference call. Confirm the
layout against the real unzipped export before trusting this end to end,
same as we did with Metrica's CSV headers.

Usage (once you have a real export):
    uv run python -m sportsgraph.detection.run_baseline_eval \
        --images-dir data/raw/warmup_labeled/images \
        --labels-dir data/raw/warmup_labeled/labels \
        --classes-file data/raw/warmup_labeled/classes.txt \
        --model yolov8n.pt
"""

import argparse
from pathlib import Path

from sportsgraph.detection.evaluate import Prediction, compute_map, match_image
from sportsgraph.detection.split import time_based_split
from sportsgraph.detection.yolo_labels import image_size, load_class_names, load_yolo_label_file

# Our label classes, mapped onto the COCO class names a pretrained model actually outputs.
OUR_CLASS_TO_COCO_NAME = {
    "player": "person",
    "goalkeeper": "person",
    "referee": "person",
    "ball": "sports ball",
}


def evaluate_split(
    image_paths: list[Path],
    labels_dir: Path,
    class_names: dict[int, str],
    model,
    coco_id_by_our_class: dict[str, int],
    iou_threshold: float,
) -> dict[str, float]:
    matches_by_class: dict[str, list[tuple[float, bool]]] = {c: [] for c in coco_id_by_our_class}
    num_gt_by_class: dict[str, int] = dict.fromkeys(coco_id_by_our_class, 0)

    for image_path in image_paths:
        img_w, img_h = image_size(image_path)
        label_path = labels_dir / f"{image_path.stem}.txt"
        ground_truths = load_yolo_label_file(label_path, class_names, img_w, img_h)

        result = model.predict(str(image_path), conf=0.001, verbose=False)[0]

        for our_class, coco_id in coco_id_by_our_class.items():
            our_class_id = next(cid for cid, name in class_names.items() if name == our_class)
            gt_for_class = [g for g in ground_truths if g.class_id == our_class_id]
            num_gt_by_class[our_class] += len(gt_for_class)

            preds_for_class = [
                Prediction(
                    box=tuple(box.xyxy[0].tolist()),
                    confidence=float(box.conf[0]),
                    class_id=our_class_id,
                )
                for box in result.boxes
                if int(box.cls[0]) == coco_id
            ]
            matches_by_class[our_class].extend(
                match_image(preds_for_class, gt_for_class, iou_threshold)
            )

    return compute_map(matches_by_class, num_gt_by_class)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--classes-file", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--held-out-frac", type=float, default=0.2)
    args = parser.parse_args()

    from ultralytics import YOLO

    class_names = load_class_names(args.classes_file)
    image_paths = sorted(args.images_dir.glob("*.jpg")) + sorted(args.images_dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"No images found in {args.images_dir}")

    working, held_out = time_based_split(image_paths, args.held_out_frac)
    print(f"{len(working)} working frames, {len(held_out)} held-out frames")

    model = YOLO(args.model)
    model_name_to_id = {name: cid for cid, name in model.names.items()}
    coco_id_by_our_class = {}
    for our_class in class_names.values():
        if our_class in model_name_to_id:
            coco_id_by_our_class[our_class] = model_name_to_id[our_class]
        elif our_class in OUR_CLASS_TO_COCO_NAME:
            coco_name = OUR_CLASS_TO_COCO_NAME[our_class]
            if coco_name in model_name_to_id:
                coco_id_by_our_class[our_class] = model_name_to_id[coco_name]

    for split_name, paths in [("working", working), ("held-out", held_out)]:
        result = evaluate_split(
            paths, args.labels_dir, class_names, model, coco_id_by_our_class, args.iou_threshold
        )
        print(f"\n{split_name} set:")
        for class_name, ap in result.items():
            print(f"  {class_name}: {ap:.4f}")


if __name__ == "__main__":
    main()
