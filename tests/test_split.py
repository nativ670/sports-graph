from pathlib import Path

import pytest

from sportsgraph.detection.split import time_based_split


def test_split_keeps_time_order_held_out_is_the_latest_frames():
    paths = [Path(f"frame_{i:04d}.jpg") for i in range(10)]
    working, held_out = time_based_split(paths, held_out_frac=0.2)
    assert [p.name for p in working] == [f"frame_{i:04d}.jpg" for i in range(8)]
    assert [p.name for p in held_out] == ["frame_0008.jpg", "frame_0009.jpg"]


def test_split_is_order_independent_input():
    # Shuffled input order shouldn't matter -- the function sorts internally.
    paths = [
        Path("frame_0003.jpg"),
        Path("frame_0001.jpg"),
        Path("frame_0002.jpg"),
        Path("frame_0000.jpg"),
    ]
    working, held_out = time_based_split(paths, held_out_frac=0.25)
    assert [p.name for p in working] == ["frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg"]
    assert [p.name for p in held_out] == ["frame_0003.jpg"]


def test_tiny_input_still_gets_a_nonempty_held_out_set():
    paths = [Path("frame_0000.jpg"), Path("frame_0001.jpg")]
    working, held_out = time_based_split(paths, held_out_frac=0.1)
    assert len(working) >= 1
    assert len(held_out) >= 1


def test_invalid_fraction_raises():
    with pytest.raises(ValueError):
        time_based_split([Path("a.jpg")], held_out_frac=1.5)
