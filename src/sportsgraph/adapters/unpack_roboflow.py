"""Unpack the downloaded Roboflow YOLO dataset zip into data/raw/warmup_labeled."""

import zipfile
from pathlib import Path

import yaml


def unpack_dataset(
    zip_path: Path = Path("data/raw/football-players-detection.v20-rf-detr-m.yolov11.zip"),
    target_dir: Path = Path("data/raw/warmup_labeled"),
    split: str = "all",
) -> None:
    """Unpack images and labels from the Roboflow dataset into a standardized layout."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found at {zip_path}")

    target_images = target_dir / "images"
    target_labels = target_dir / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

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
        extracted_images = 0
        extracted_labels = 0

        for member in z.namelist():
            for s in splits_to_extract:
                if member.startswith(f"{s}/images/") and member.lower().endswith(
                    (".jpg", ".png", ".jpeg")
                ):
                    filename = Path(member).name
                    dest_img = target_images / filename
                    dest_img.write_bytes(z.read(member))
                    extracted_images += 1

                    # Look for corresponding label file
                    label_member = f"{s}/labels/{Path(member).stem}.txt"
                    if label_member in z.namelist():
                        dest_label = target_labels / f"{Path(member).stem}.txt"
                        dest_label.write_bytes(z.read(label_member))
                        extracted_labels += 1
                    else:
                        # Frame with no annotations -> empty text file
                        (target_labels / f"{Path(member).stem}.txt").write_text("")
                        extracted_labels += 1

        print(
            f"Successfully unpacked {extracted_images} images and "
            f"{extracted_labels} label files into {target_dir}"
        )


if __name__ == "__main__":
    unpack_dataset()
