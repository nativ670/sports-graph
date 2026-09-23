import itertools

import networkx as nx
import pandas as pd
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.distance import pdist, squareform


def build_frame_graph(
    df: pd.DataFrame,
    frame: int,
    team: str,
    edge_rule: str = "delaunay",
    radius_m: float = 15.0,
) -> nx.Graph:
    """Build a graph of one team's players in one frame.

    Nodes are track_ids, carrying position as a node attribute.
    edge_rule: "delaunay" or "radius".
    """
    if edge_rule not in ["delaunay", "radius"]:
        raise ValueError("edge_rule must be 'delaunay' or 'radius'")

    G = nx.Graph()
    mask = (df["frame"] == frame) & (df["team"] == team)
    f_df = df[mask]

    if len(f_df) == 0:
        return G

    positions = []
    track_ids = []

    for _, row in f_df.iterrows():
        track_id = int(row["track_id"])
        x, y = float(row["x_m"]), float(row["y_m"])
        G.add_node(track_id, pos=(x, y))
        positions.append([x, y])
        track_ids.append(track_id)

    if len(f_df) < 2:
        return G

    if edge_rule == "radius":
        dist_matrix = squareform(pdist(positions))
        n = len(track_ids)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix[i, j] <= radius_m:
                    G.add_edge(track_ids[i], track_ids[j])

    elif edge_rule == "delaunay":
        if len(f_df) < 3:
            return G

        try:
            tri = Delaunay(positions)
            for simplex in tri.simplices:
                pairs = itertools.combinations(simplex, 2)
                for i, j in pairs:
                    G.add_edge(track_ids[i], track_ids[j])
        except QhullError:
            # Fall back to no edges if points are collinear or Qhull fails
            pass

    return G
