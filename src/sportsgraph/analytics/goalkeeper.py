import pandas as pd

from sportsgraph.contract.meta import MatchMeta


def identify_goalkeeper(df: pd.DataFrame, meta: MatchMeta, team: str) -> int:
    """
    Identifies the goalkeeper for a given team by finding the track_id with the
    lowest positional variance (std of x and y) consistently across periods.
    """
    team_df = df[df["team"] == team]
    if team_df.empty:
        raise ValueError(f"No tracking data found for team '{team}'.")

    boundaries = sorted(list(meta.period_boundary_frames))

    # Assign periods to frames
    def get_period(frame: int) -> int:
        period = 1
        for b in boundaries:
            if frame >= b:
                period += 1
            else:
                break
        return period

    team_df = team_df.assign(period=team_df["frame"].apply(get_period))

    # Calculate std per period and track_id
    stats = team_df.groupby(["period", "track_id"])[["x_m", "y_m"]].std().reset_index()
    # Combine x and y standard deviations into a single "spread" metric (e.g., area or sum of stds)
    stats["spread"] = stats["x_m"] * stats["y_m"]

    # Find the track_id with the consistently lowest spread across periods
    # Average the spread across periods to find the most consistently stationary player
    avg_spread = stats.groupby("track_id")["spread"].mean()

    gk_track_id = avg_spread.idxmin()
    return int(gk_track_id)
