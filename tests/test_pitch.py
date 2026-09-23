import matplotlib.pyplot as plt
import pandas as pd

from sportsgraph.analytics.pitch import plot_trail


def test_plot_trail_markers():
    """
    Ensure plot_trail correctly places the start and end markers
    on the first and last coordinates of a player's slice.
    """
    df = pd.DataFrame(
        {
            "frame": [1, 2, 3],
            "track_id": [9, 9, 9],
            "team": ["home", "home", "home"],
            "x_m": [10.0, 15.0, 20.0],
            "y_m": [5.0, 10.0, 15.0],
        }
    )

    fig, ax = plt.subplots()
    plot_trail(df, track_id=9, frame_start=1, frame_end=3, ax=ax)

    # plot_trail adds 3 objects to an empty ax: the trail line, start marker, end marker
    lines = ax.lines
    assert len(lines) == 3

    trail, start_marker, end_marker = lines[0], lines[1], lines[2]

    # Check start point
    assert start_marker.get_xdata()[0] == 10.0
    assert start_marker.get_ydata()[0] == 5.0

    # Check end point
    assert end_marker.get_xdata()[0] == 20.0
    assert end_marker.get_ydata()[0] == 15.0

    # Check trail line arrays
    assert list(trail.get_xdata()) == [10.0, 15.0, 20.0]
    assert list(trail.get_ydata()) == [5.0, 10.0, 15.0]
