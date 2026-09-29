import numpy as np
import pandas as pd

from sportsgraph.analytics.graph import build_frame_graph
from sportsgraph.analytics.metrics import compute_frame_metrics


def build_shuffled_frame(
    f_df: pd.DataFrame, player_positions: dict[int, np.ndarray], rng: np.random.Generator
) -> pd.DataFrame:
    """
    Given a real frame (f_df) and a dictionary of all historical positions per track_id,
    replace each player's position with a randomly chosen one from their history.
    """
    shuffled_df = f_df.copy()

    # We iterate over the index so we can safely assign back to the copy
    for idx, row in f_df.iterrows():
        track_id = row["track_id"]
        positions = player_positions.get(track_id)

        if positions is not None and len(positions) > 1:
            # Pick a random position. (With 100,000+ frames, the chance of picking
            # the exact same frame is negligible, so we skip the expensive exclusion check).
            rand_idx = rng.integers(0, len(positions))
            shuffled_df.at[idx, "x_m"] = positions[rand_idx, 0]
            shuffled_df.at[idx, "y_m"] = positions[rand_idx, 1]

    return shuffled_df


def compare_real_vs_shuffled(
    df: pd.DataFrame, team: str, edge_rule: str, n_samples: int, seed: int
) -> pd.DataFrame:
    """
    Sample n_samples real frames for `team`. For each, compute metrics on the
    real frame and on one shuffled surrogate. Return one long table with a
    `kind` column ("real" / "shuffled") so the two distributions are easy to
    compare side by side.

    Note for statistical testing (e.g. Mann-Whitney U): When edge rules like
    'radius' cause the shuffled graph to almost always disconnect, some metrics
    like algebraic connectivity may have zero or near-zero variance. Rank-based
    tests on constant distributions should be interpreted cautiously.
    """
    rng = np.random.default_rng(seed)

    team_df = df[df["team"] == team]

    # Pre-compute positions as fast numpy arrays to avoid slow Pandas filtering in loops
    player_positions = {
        track_id: group[["x_m", "y_m"]].values for track_id, group in team_df.groupby("track_id")
    }

    # Group frames once
    frames_gb = team_df.groupby("frame")
    valid_frames = [f for f, group in frames_gb if len(group) > 1]

    if len(valid_frames) < n_samples:
        sampled_frames = valid_frames
    else:
        sampled_frames = rng.choice(valid_frames, size=n_samples, replace=False)

    results = []
    for f in sampled_frames:
        f_df = frames_gb.get_group(f)

        # Real frame
        G_real = build_frame_graph(f_df, frame=f, team=team, edge_rule=edge_rule)
        real_metrics = compute_frame_metrics(G_real)
        real_metrics["frame"] = f
        real_metrics["kind"] = "real"
        results.append(real_metrics)

        # Shuffled frame
        shuffled_df = build_shuffled_frame(f_df, player_positions, rng)
        G_shuff = build_frame_graph(shuffled_df, frame=f, team=team, edge_rule=edge_rule)
        shuff_metrics = compute_frame_metrics(G_shuff)
        shuff_metrics["frame"] = f
        shuff_metrics["kind"] = "shuffled"
        results.append(shuff_metrics)

    return pd.DataFrame(results)
