"""Unpack the downloaded Roboflow YOLO dataset zip into data/raw/warmup_labeled."""

import zipfile
from pathlib import Path

import yaml


def unpack_dataset(
    zip_path: Path = Path("data/raw/football-players-detection.v20-rf-detr-m.yolov11.zip"),
    target_dir: Path = Path("data/raw/warmup_labeled"),
    split: str = "all",
    preserve_splits: bool = True,
) -> dict[str, int]:
    """Unpack images and labels from the Roboflow dataset into a standardized layout.

    When preserve_splits is True (default for Roboflow), creates native train/, valid/,
    and test/ subdirectories each with images/ and labels/.
    When preserve_splits is False (flat layout for CVAT compatibility), places all
    images and labels directly under target_dir/images and target_dir/labels.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found at {zip_path}")

    target_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    with zipfile.ZipFile(zip_path, "r") as z:
        # 1. Parse data.yaml to extract class names
        if "data.yaml" in z.namelist():
            data_yaml = yaml.safe_load(z.read("data.yaml").decode("utf-8"))
            class_names = data_yaml.get("names", [])
        else:
            class_names = ["ball", "goalkeeper", "player", "referee"]

        classes_file = target_dir / "classes.txt"
        classes_file.write_text("\n".join(class_names) + "\n", encoding="utf-8")
        print(f"Wrote classes.txt with classes: {class_names}")

        # 2. Extract images and labels
        splits_to_extract = ["train", "valid", "test"] if split == "all" else [split]

        for s in splits_to_extract:
            if preserve_splits:
                s_images = target_dir / s / "images"
                s_labels = target_dir / s / "labels"
            else:
                s_images = target_dir / "images"
                s_labels = target_dir / "labels"

            s_images.mkdir(parents=True, exist_ok=True)
            s_labels.mkdir(parents=True, exist_ok=True)
            extracted_split_images = 0

            for member in z.namelist():
                if member.startswith(f"{s}/images/") and member.lower().endswith(
                    (".jpg", ".png", ".jpeg")
                ):
                    filename = Path(member).name
                    dest_img = s_images / filename
                    dest_img.write_bytes(z.read(member))
                    extracted_split_images += 1

                    # Look for corresponding label file
                    label_member = f"{s}/labels/{Path(member).stem}.txt"
                    dest_label = s_labels / f"{Path(member).stem}.txt"
                    if label_member in z.namelist():
                        dest_label.write_bytes(z.read(label_member))
                    else:
                        dest_label.write_text("")

            counts[s] = extracted_split_images
            if extracted_split_images > 0:
                print(f"Split '{s}': unpacked {extracted_split_images} frames into {s_images}")

    total_images = sum(counts.values())
    print(f"Successfully unpacked {total_images} total images into {target_dir}")
    return counts


if __name__ == "__main__":
    unpack_dataset()
