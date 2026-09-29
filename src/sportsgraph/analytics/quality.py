import numpy as np
import pandas as pd

from sportsgraph.contract.meta import MatchMeta


def flag_implausible_speed(
    df: pd.DataFrame, meta: MatchMeta, max_speed_mps: float = 12.0
) -> pd.DataFrame:
    """Flag (frame, team, track_id) rows where implied speed since the player's
    previous observation exceeds max_speed_mps.

    Excludes exact frames that mark period boundaries (e.g. half-time restarts),
    since that's a data-representation issue, not a per-player tracking failure.
    """
    df = df.sort_values(["team", "track_id", "frame"]).copy()
    g = df.groupby(["team", "track_id"])
    dist = np.hypot(df["x_m"] - g["x_m"].shift(1), df["y_m"] - g["y_m"].shift(1))
    elapsed_s = (df["frame"] - g["frame"].shift(1)) / meta.fps
    df["speed_mps"] = dist / elapsed_s

    # Ignore jump if the frame is exactly at a known period boundary
    is_boundary = df["frame"].isin(meta.period_boundary_frames)

    mask = (df["speed_mps"] > max_speed_mps) & (~is_boundary)
    return df.loc[mask, ["frame", "team", "track_id", "speed_mps"]]
