import pandas as pd

from sportsgraph.analytics.goalkeeper import identify_goalkeeper
from sportsgraph.contract.meta import MatchMeta


def test_identify_goalkeeper():
    """
    Test that identify_goalkeeper correctly finds the track_id with the lowest
    positional variance within periods.
    """
    # Create 3 players across 2 periods (frames 1-5, 6-10)
    # Player 1: Very high variance (outfielder)
    # Player 2: Low variance (goalkeeper)
    # Player 3: Medium variance (outfielder)

    records = []

    # Period 1 (frames 1-5)
    for frame in range(1, 6):
        # Player 1: Moves 10m every frame
        records.append(
            {"frame": frame, "track_id": 1, "team": "home", "x_m": frame * 10, "y_m": frame * 10}
        )
        # Player 2: Moves 1m every frame (low variance)
        records.append(
            {"frame": frame, "track_id": 2, "team": "home", "x_m": -40 + frame, "y_m": frame}
        )
        # Player 3: Moves 5m every frame
        records.append(
            {"frame": frame, "track_id": 3, "team": "home", "x_m": frame * 5, "y_m": frame * 5}
        )

    # Period 2 (frames 6-10)
    for frame in range(6, 11):
        # Player 1: Moves 10m every frame
        records.append(
            {"frame": frame, "track_id": 1, "team": "home", "x_m": frame * 10, "y_m": frame * 10}
        )
        # Player 2: Moves 1m every frame but on the other side of the pitch
        records.append(
            {"frame": frame, "track_id": 2, "team": "home", "x_m": 40 + frame, "y_m": frame}
        )
        # Player 3: Moves 5m every frame
        records.append(
            {"frame": frame, "track_id": 3, "team": "home", "x_m": frame * 5, "y_m": frame * 5}
        )

    df = pd.DataFrame(records)

    meta = MatchMeta(
        fps=25.0,
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        period_boundary_frames=(6,),  # Period 2 starts at frame 6
    )

    gk_id = identify_goalkeeper(df, meta, "home")

    assert gk_id == 2
