"""Train/held-out split by time, not at random.

Frames pulled from one continuous clip are near-duplicates of their
neighbors, so a random split leaks information between the two sets (the
same leakage problem as randomly splitting adjacent video frames in a
tracking table). Splitting by position in the sequence keeps the held-out
set genuinely unseen.
"""

from pathlib import Path


def time_based_split(
    image_paths: list[Path], held_out_frac: float = 0.2
) -> tuple[list[Path], list[Path]]:
    """Sort frames by filename, then take the LAST `held_out_frac` as held-out.

    Assumes filenames sort into true time order (e.g. frame_0001.jpg,
    frame_0002.jpg, ...), which is how CVAT and ffmpeg both name sequential
    frame exports by default.
    """
    if not 0.0 < held_out_frac < 1.0:
        raise ValueError("held_out_frac must be between 0 and 1")

    ordered = sorted(image_paths, key=lambda p: p.name)
    split_idx = int(len(ordered) * (1 - held_out_frac))
    # Guarantee at least one frame in each half for any non-trivial input,
    # rather than silently returning an empty held-out set on a small input.
    split_idx = max(1, min(split_idx, len(ordered) - 1))
    return ordered[:split_idx], ordered[split_idx:]
