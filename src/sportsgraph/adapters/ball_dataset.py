"""Utilities for extracting and combining soccer ball annotations into a single-class dataset."""

import zipfile
from pathlib import Path


def filter_ball_label_content(content: str, ball_class_id: int = 0) -> str:
    """Filter YOLO annotation text to keep only the ball class, remapping its ID to 0.

    Lines with classes other than `ball_class_id` are removed.
    Remaining lines are remapped to class 0.
    """
    filtered_lines = []
    for line in content.strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cid = int(parts[0])
            if cid == ball_class_id:
                # Remap to class 0 for single-class ball dataset
                remapped_line = " ".join(["0"] + parts[1:])
                filtered_lines.append(remapped_line)
        except ValueError:
            continue

    return "\n".join(filtered_lines) + ("\n" if filtered_lines else "")


def filter_ball_annotations(
    source_labels_dir: Path,
    target_labels_dir: Path,
    ball_class_id: int = 0,
) -> int:
    """Filter an entire directory of YOLO label files, writing ball-only labels.

    Frames with no ball annotations produce empty text files (valid YOLO background images).
    Returns the count of processed label files.
    """
    target_labels_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for label_file in sorted(source_labels_dir.glob("*.txt")):
        if label_file.name == "classes.txt":
            continue
        filtered_content = filter_ball_label_content(
            label_file.read_text(encoding="utf-8"), ball_class_id=ball_class_id
        )
        dest_file = target_labels_dir / label_file.name
        dest_file.write_text(filtered_content, encoding="utf-8")
        count += 1
    return count


def unpack_and_filter_players_zip(
    zip_path: Path,
    target_dir: Path,
    ball_class_id: int = 0,
    split: str = "train",
) -> tuple[int, int]:
    """Extract frames and filter labels to ball-only from a multi-class Roboflow zip.

    Returns:
        tuple[int, int]: (images_extracted, labels_extracted)
    """
    images_dir = target_dir / "images"
    labels_dir = target_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    extracted_images = 0
    extracted_labels = 0

    with zipfile.ZipFile(zip_path, "r") as z:
        splits = ["train", "valid", "test"] if split == "all" else [split]
        for member in z.namelist():
            for s in splits:
                if member.startswith(f"{s}/images/") and member.lower().endswith(
                    (".jpg", ".png", ".jpeg")
                ):
                    dest_img = images_dir / Path(member).name
                    dest_img.write_bytes(z.read(member))
                    extracted_images += 1

                    # Look for label file
                    label_member = f"{s}/labels/{Path(member).stem}.txt"
                    dest_label = labels_dir / f"{Path(member).stem}.txt"
                    if label_member in z.namelist():
                        raw_content = z.read(label_member).decode("utf-8")
                        dest_label.write_text(
                            filter_ball_label_content(raw_content, ball_class_id),
                            encoding="utf-8",
                        )
                    else:
                        dest_label.write_text("", encoding="utf-8")
                    extracted_labels += 1

    return extracted_images, extracted_labels
