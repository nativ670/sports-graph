"""Fine-tune YOLO detector on soccer dataset with transfer learning and augmentations.

Supports both local execution (CPU / local GPU) and remote execution (e.g. Kaggle / Colab).
"""

import argparse
from pathlib import Path

import yaml

from sportsgraph.detection.split import time_based_split
from sportsgraph.detection.yolo_labels import load_class_names


def prepare_training_manifest(dataset_dir: Path, held_out_frac: float = 0.2) -> Path:
    """Prepare train.txt, val.txt, and a dataset YAML manifest for YOLO training.

    If native Roboflow split subdirectories exist (train/images and valid/images),
    they are used directly to guarantee clean, zero-leakage training.
    Otherwise (e.g. flat CVAT video export), uses our deterministic time_based_split.
    """
    dataset_dir = dataset_dir.resolve()
    classes_file = dataset_dir / "classes.txt"

    if not classes_file.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_file}")

    class_names_dict = load_class_names(classes_file)
    class_names = [class_names_dict[i] for i in range(len(class_names_dict))]

    train_split_dir = dataset_dir / "train" / "images"
    valid_split_dir = dataset_dir / "valid" / "images"

    if train_split_dir.exists() and valid_split_dir.exists():
        working_paths = sorted(train_split_dir.glob("*.jpg")) + sorted(
            train_split_dir.glob("*.png")
        )
        held_out_paths = sorted(valid_split_dir.glob("*.jpg")) + sorted(
            valid_split_dir.glob("*.png")
        )
        if not working_paths:
            raise ValueError(f"No training images found in {train_split_dir}")
        if not held_out_paths:
            raise ValueError(f"No validation images found in {valid_split_dir}")
    else:
        images_dir = dataset_dir / "images"
        if not images_dir.exists():
            msg = (
                f"Neither split directories (train/valid) nor flat images "
                f"directory found in {dataset_dir}"
            )
            raise FileNotFoundError(msg)
        image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        if not image_paths:
            raise ValueError(f"No images found in {images_dir}")
        working_paths, held_out_paths = time_based_split(image_paths, held_out_frac=held_out_frac)

    train_txt = dataset_dir / "train.txt"
    val_txt = dataset_dir / "val.txt"

    train_txt.write_text("\n".join(str(p.resolve().as_posix()) for p in working_paths) + "\n")
    val_txt.write_text("\n".join(str(p.resolve().as_posix()) for p in held_out_paths) + "\n")

    manifest = {
        "path": str(dataset_dir.as_posix()),
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(class_names),
        "names": class_names,
    }

    test_split_dir = dataset_dir / "test" / "images"
    if test_split_dir.exists():
        test_paths = sorted(test_split_dir.glob("*.jpg")) + sorted(test_split_dir.glob("*.png"))
        if test_paths:
            test_txt = dataset_dir / "test.txt"
            test_txt.write_text("\n".join(str(p.resolve().as_posix()) for p in test_paths) + "\n")
            manifest["test"] = "test.txt"

    manifest_path = dataset_dir / "dataset.yaml"
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(
        f"Manifest prepared: {len(working_paths)} train frames, "
        f"{len(held_out_paths)} val frames -> {manifest_path}"
    )
    return manifest_path


def train_model(
    dataset_yaml: Path,
    model_name: str = "yolo11n.pt",
    epochs: int = 50,
    imgsz: int = 1280,
    batch: int = 8,
    device: str | None = None,
    output_dir: Path = Path("models/fine_tuned"),
):
    """Fine-tune YOLO using transfer learning and regularizing augmentations."""
    from ultralytics import YOLO

    model = YOLO(model_name)

    # Configure training with data augmentations acting as regularization
    train_args = {
        "data": str(dataset_yaml.resolve().as_posix()),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(output_dir.parent.as_posix()),
        "name": output_dir.name,
        "exist_ok": True,
        # Regularizing augmentations
        "mosaic": 1.0,
        "mixup": 0.15,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 0.5,
        "verbose": True,
    }

    if device:
        train_args["device"] = device

    print(f"Starting training on {dataset_yaml} with {model_name}...")
    results = model.train(**train_args)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO detector on soccer dataset")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/raw/warmup_labeled"),
        help="Path to dataset containing images/, labels/, and classes.txt",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Base model checkpoint for transfer learning",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (e.g. '0', 'cpu', or leave empty for auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/fine_tuned"),
        help="Destination directory for trained weights",
    )
    parser.add_argument(
        "--held-out-frac",
        type=float,
        default=0.2,
        help="Fraction of latest frames reserved for validation",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate the dataset.yaml and split files without launching training",
    )
    args = parser.parse_args()

    dataset_yaml = prepare_training_manifest(args.dataset_dir, held_out_frac=args.held_out_frac)

    if not args.prepare_only:
        train_model(
            dataset_yaml=dataset_yaml,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
