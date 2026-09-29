import pandas as pd

from sportsgraph.analytics.quality import flag_implausible_speed
from sportsgraph.contract.meta import MatchMeta


def test_flag_implausible_speed_distinguishes_glitch_from_period_boundary():
    """
    Test that a single player teleporting (glitch) is flagged,
    but a teleport that matches a known period boundary is ignored.
    """
    # At 25 fps, 1 frame = 0.04s. Max speed 12m/s = max dist 0.48m per frame.
    data = [
        # Frame 1: Base positions
        (1, "home", 1, 0.0, 0.0),
        (1, "home", 2, 0.0, 0.0),
        # Frame 2: P1 glitches (moves 10m -> 250 m/s).
        # This is not a period boundary, so it should be flagged.
        (2, "home", 1, 10.0, 0.0),
        (2, "home", 2, 0.0, 0.0),
        # Frame 3: Mass jump (teleport to second half).
        # Everyone moves 50m in 1 frame -> 1250 m/s.
        # This frame is marked in meta as a period boundary, so it is ignored.
        (3, "home", 1, 60.0, 0.0),
        (3, "home", 2, 50.0, 0.0),
    ]

    df = pd.DataFrame(data, columns=["frame", "team", "track_id", "x_m", "y_m"])
    meta = MatchMeta(
        fps=25.0, pitch_length_m=105.0, pitch_width_m=68.0, period_boundary_frames=(3,)
    )

    flagged = flag_implausible_speed(df, meta=meta, max_speed_mps=12.0)

    # Assert only P1 in frame 2 is flagged
    assert len(flagged) == 1
    assert flagged.iloc[0]["frame"] == 2
    assert flagged.iloc[0]["track_id"] == 1
    assert flagged.iloc[0]["team"] == "home"
    assert flagged.iloc[0]["speed_mps"] == 250.0
