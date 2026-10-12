"""Unit tests for soccer ball dataset filtering and adapter."""

import zipfile
from pathlib import Path

from sportsgraph.adapters.ball_dataset import (
    filter_ball_annotations,
    filter_ball_label_content,
    unpack_and_filter_players_zip,
)


def test_filter_ball_label_content():
    content = (
        "0 0.5 0.5 0.02 0.02\n"  # ball
        "1 0.1 0.5 0.04 0.12\n"  # goalkeeper
        "2 0.4 0.6 0.05 0.11\n"  # player
        "3 0.8 0.5 0.04 0.10\n"  # referee
    )
    result = filter_ball_label_content(content, ball_class_id=0)
    assert result == "0 0.5 0.5 0.02 0.02\n"

    # Test when no ball is present
    no_ball = "2 0.4 0.6 0.05 0.11\n3 0.8 0.5 0.04 0.10\n"
    assert filter_ball_label_content(no_ball, ball_class_id=0) == ""


def test_filter_ball_annotations_dir(tmp_path: Path):
    source_dir = tmp_path / "source_labels"
    target_dir = tmp_path / "target_labels"
    source_dir.mkdir()

    (source_dir / "classes.txt").write_text("ball\ngoalkeeper\nplayer\nreferee\n")
    (source_dir / "frame_0001.txt").write_text("0 0.5 0.5 0.02 0.02\n2 0.4 0.4 0.1 0.1\n")
    (source_dir / "frame_0002.txt").write_text("2 0.3 0.3 0.1 0.1\n")

    count = filter_ball_annotations(source_dir, target_dir, ball_class_id=0)
    assert count == 2

    assert (target_dir / "frame_0001.txt").read_text() == "0 0.5 0.5 0.02 0.02\n"
    assert (target_dir / "frame_0002.txt").read_text() == ""
    assert not (target_dir / "classes.txt").exists()


def test_unpack_and_filter_players_zip(tmp_path: Path):
    zip_path = tmp_path / "test_players.zip"
    target_dir = tmp_path / "extracted"

    # Build dummy zip
    with zipfile.ZipFile(zip_path, "w") as z:
        img_bytes = b"fake-jpg-data"
        z.writestr("train/images/frame_001.jpg", img_bytes)
        z.writestr("train/images/frame_002.jpg", img_bytes)
        z.writestr("train/labels/frame_001.txt", "0 0.5 0.5 0.01 0.01\n2 0.2 0.2 0.1 0.1\n")
        z.writestr("train/labels/frame_002.txt", "2 0.2 0.2 0.1 0.1\n")

    imgs, labels = unpack_and_filter_players_zip(
        zip_path, target_dir, ball_class_id=0, split="train"
    )

    assert imgs == 2
    assert labels == 2
    assert (target_dir / "images" / "frame_001.jpg").exists()
    assert (target_dir / "labels" / "frame_001.txt").read_text() == "0 0.5 0.5 0.01 0.01\n"
    assert (target_dir / "labels" / "frame_002.txt").read_text() == ""
