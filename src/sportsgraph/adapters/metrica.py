from pathlib import Path

import pandas as pd

from sportsgraph.contract.meta import MatchMeta
from sportsgraph.contract.schema import validate_players

METRICA_LENGTH = 105.0
METRICA_WIDTH = 68.0


def load_metrica_tracking(filepath: Path | str, team: str) -> pd.DataFrame:
    """
    Load Metrica tracking CSV and convert it to the standardized long format.

    This adapter handles a known Metrica bug where substitution placeholders
    cause players coming on/off to share identical coordinates for a single frame;
    both duplicate rows are aggressively dropped to preserve data integrity.
    """
    # Header is at row index 2
    df = pd.read_csv(filepath, header=2)

    new_cols = []
    current_player = None
    for col in df.columns:
        col_str = str(col)
        if col_str.startswith("Player"):
            current_player = col_str.replace("Player", "")
            new_cols.append(f"x_{current_player}")
        elif col_str.startswith("Unnamed") and current_player is not None:
            new_cols.append(f"y_{current_player}")
            current_player = None
        else:
            # Note: Ball columns are intentionally left alone here; this function only loads players
            new_cols.append(col_str)
            current_player = None

    df.columns = new_cols
    df = df.rename(columns={"Frame": "frame"})

    # Reshape from wide to long
    long_df = pd.wide_to_long(df, stubnames=["x", "y"], i="frame", j="track_id", sep="_")
    long_df = long_df.reset_index()

    # Drop rows where coordinates are NaN
    long_df = long_df.dropna(subset=["x", "y"])

    # Add team identifier
    long_df["team"] = team

    # Convert units: [0, 1] to meters from pitch center
    # Contract says: origin at pitch center, x along length, y up.
    # Metrica origin is top-left, so x grows right, y grows down.
    long_df["x_m"] = (long_df["x"] - 0.5) * METRICA_LENGTH
    long_df["y_m"] = (0.5 - long_df["y"]) * METRICA_WIDTH

    # Cast identifiers
    long_df["frame"] = long_df["frame"].astype(int)
    long_df["track_id"] = long_df["track_id"].astype(int)

    # Select columns as per contract
    final_df = long_df[["frame", "track_id", "team", "x_m", "y_m"]]

    # Metrica substitution placeholder bug: player coming off and on share exact coordinates
    # for a single frame, resulting in 12 players. We drop both duplicate rows at that frame.
    final_df = final_df.drop_duplicates(subset=["frame", "team", "x_m", "y_m"], keep=False)

    # Final sort
    final_df = final_df.sort_values(["frame", "track_id"]).reset_index(drop=True)

    # Metrica sample data is 25 fps
    meta = MatchMeta(fps=25.0, pitch_length_m=METRICA_LENGTH, pitch_width_m=METRICA_WIDTH)

    return validate_players(final_df, meta)
