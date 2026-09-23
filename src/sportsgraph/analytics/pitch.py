import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Arc, Circle, Rectangle


def draw_pitch(ax=None):
    """
    Draw a 105x68 pitch on the given matplotlib axis.
    The pitch is centered at the origin (touchlines at +/- 34, goal lines at +/- 52.5).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 6.8))
        fig.patch.set_facecolor("#13261c")

    half_length = 105.0 / 2
    half_width = 68.0 / 2

    ax.set_xlim(-half_length - 5, half_length + 5)
    ax.set_ylim(-half_width - 5, half_width + 5)

    # Modern, fresh styling
    pitch_color = "#1c3829"
    line_color = "#ffffff"
    line_alpha = 0.6
    line_width = 1.5

    ax.set_facecolor(pitch_color)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")

    # Pitch Outline & Centre Line
    pitch_outline = Rectangle(
        (-half_length, -half_width),
        105.0,
        68.0,
        fill=False,
        color=line_color,
        linewidth=line_width,
        alpha=line_alpha,
        zorder=0,
    )
    ax.add_patch(pitch_outline)
    ax.plot(
        [0, 0],
        [-half_width, half_width],
        color=line_color,
        linewidth=line_width,
        alpha=line_alpha,
        zorder=0,
    )

    # Centre Circle (radius 9.15m)
    center_circle = Circle(
        (0, 0), 9.15, fill=False, color=line_color, linewidth=line_width, alpha=line_alpha, zorder=0
    )
    ax.add_patch(center_circle)

    # Center spot
    ax.plot(0, 0, "o", color=line_color, markersize=3, alpha=line_alpha, zorder=0)

    # Penalty Areas (16.5m long, 40.32m wide)
    pen_area_length = 16.5
    pen_area_width = 40.32

    left_pen = Rectangle(
        (-half_length, -pen_area_width / 2),
        pen_area_length,
        pen_area_width,
        fill=False,
        color=line_color,
        linewidth=line_width,
        alpha=line_alpha,
        zorder=0,
    )
    right_pen = Rectangle(
        (half_length - pen_area_length, -pen_area_width / 2),
        pen_area_length,
        pen_area_width,
        fill=False,
        color=line_color,
        linewidth=line_width,
        alpha=line_alpha,
        zorder=0,
    )
    ax.add_patch(left_pen)
    ax.add_patch(right_pen)

    # Goal Areas (5.5m long, 18.32m wide)
    goal_area_length = 5.5
    goal_area_width = 18.32

    left_goal_area = Rectangle(
        (-half_length, -goal_area_width / 2),
        goal_area_length,
        goal_area_width,
        fill=False,
        color=line_color,
        linewidth=line_width,
        alpha=line_alpha,
        zorder=0,
    )
    right_goal_area = Rectangle(
        (half_length - goal_area_length, -goal_area_width / 2),
        goal_area_length,
        goal_area_width,
        fill=False,
        color=line_color,
        linewidth=line_width,
        alpha=line_alpha,
        zorder=0,
    )
    ax.add_patch(left_goal_area)
    ax.add_patch(right_goal_area)

    # Penalty spots (11m from goal line)
    ax.plot(-half_length + 11.0, 0, "o", color=line_color, markersize=3, alpha=line_alpha, zorder=0)
    ax.plot(half_length - 11.0, 0, "o", color=line_color, markersize=3, alpha=line_alpha, zorder=0)

    # Penalty arcs (D). Radius 9.15m from penalty spot.
    left_arc = Arc(
        (-half_length + 11.0, 0),
        width=9.15 * 2,
        height=9.15 * 2,
        angle=0,
        theta1=-53.0,
        theta2=53.0,
        color=line_color,
        linewidth=line_width,
        alpha=line_alpha,
        zorder=0,
    )
    right_arc = Arc(
        (half_length - 11.0, 0),
        width=9.15 * 2,
        height=9.15 * 2,
        angle=180,
        theta1=-53.0,
        theta2=53.0,
        color=line_color,
        linewidth=line_width,
        alpha=line_alpha,
        zorder=0,
    )
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    return ax


def plot_trail(df: pd.DataFrame, track_id: int, frame_start: int, frame_end: int, ax=None):
    """
    Plots a player's path over a given frame range on the pitch.
    """
    if ax is None:
        ax = draw_pitch()

    mask = (df["track_id"] == track_id) & (df["frame"] >= frame_start) & (df["frame"] <= frame_end)
    player_data = df[mask].sort_values("frame")

    if not player_data.empty:
        # Plot path with modern styling
        ax.plot(
            player_data["x_m"],
            player_data["y_m"],
            marker="",
            linestyle="-",
            linewidth=2.5,
            color="#fca311",
            alpha=0.85,
            zorder=3,
        )

        # Start and end points
        ax.plot(
            player_data["x_m"].iloc[0],
            player_data["y_m"].iloc[0],
            "o",
            color="#14b8a6",
            markersize=7,
            markeredgecolor="white",
            label="Start",
            zorder=4,
        )
        ax.plot(
            player_data["x_m"].iloc[-1],
            player_data["y_m"].iloc[-1],
            "o",
            color="#ef4444",
            markersize=7,
            markeredgecolor="white",
            label="End",
            zorder=4,
        )

    return ax
