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
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Root dataset directory with classes.txt and split folders or flat images/labels.",
    )
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--classes-file", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=["train", "valid", "val", "test", "all"],
        default=None,
        help="Evaluate a specific native Roboflow split directory (train, valid, or test).",
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--held-out-frac", type=float, default=0.2)
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Evaluate all images as a single set without splitting.",
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    # Resolve paths: dataset-dir vs explicit images-dir/labels-dir
    dataset_dir = args.dataset_dir
    classes_file = args.classes_file
    if classes_file is None and dataset_dir is not None:
        classes_file = dataset_dir / "classes.txt"
    if classes_file is None:
        raise SystemExit(
            "Error: --classes-file or --dataset-dir containing classes.txt is required."
        )

    class_names = load_class_names(classes_file)

    # Determine evaluation targets
    eval_tasks: list[tuple[str, list[Path], Path]] = []  # (split_label, image_paths, labels_dir)

    if dataset_dir is not None and (
        args.split is not None
        or (dataset_dir / "test").exists()
        or (dataset_dir / "valid").exists()
    ):
        # Native Roboflow split directory layout
        splits_to_eval = []
        if args.split:
            s_name = "valid" if args.split == "val" else args.split
            splits_to_eval = [s_name]
        else:
            # Default to evaluating test, then valid if available
            for cand in ["test", "valid"]:
                if (dataset_dir / cand / "images").exists():
                    splits_to_eval.append(cand)

        for s in splits_to_eval:
            s_img_dir = dataset_dir / s / "images"
            s_lbl_dir = dataset_dir / s / "labels"
            if not s_img_dir.exists():
                raise SystemExit(f"Split images directory not found: {s_img_dir}")
            s_paths = sorted(s_img_dir.glob("*.jpg")) + sorted(s_img_dir.glob("*.png"))
            if not s_paths:
                print(f"Warning: No images found for split '{s}' in {s_img_dir}")
                continue
            eval_tasks.append((s, s_paths, s_lbl_dir))
    else:
        # Flat directory layout (CVAT or custom user video)
        images_dir = args.images_dir or (dataset_dir / "images" if dataset_dir else None)
        labels_dir = args.labels_dir or (dataset_dir / "labels" if dataset_dir else None)
        if images_dir is None or labels_dir is None:
            raise SystemExit(
                "Error: provide either --dataset-dir or both --images-dir and --labels-dir."
            )

        image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        if not image_paths:
            raise SystemExit(f"No images found in {images_dir}")

        if args.no_split:
            eval_tasks.append(("all", image_paths, labels_dir))
            print(f"{len(image_paths)} total frames (no split)")
        else:
            working, held_out = time_based_split(image_paths, args.held_out_frac)
            eval_tasks.append(("working", working, labels_dir))
            eval_tasks.append(("held-out", held_out, labels_dir))
            print(
                f"{len(working)} working frames, {len(held_out)} held-out frames (time-based split)"
            )

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

    for split_name, paths, lbl_dir in eval_tasks:
        result = evaluate_split(
            paths, lbl_dir, class_names, model, coco_id_by_our_class, args.iou_threshold
        )
        print(f"\n{split_name} set ({len(paths)} frames):")
        for class_name, ap in result.items():
            print(f"  {class_name}: {ap:.4f}")


if __name__ == "__main__":
    main()
