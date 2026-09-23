from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from sportsgraph.adapters.metrica import load_metrica_match
from sportsgraph.analytics.graph import build_frame_graph
from sportsgraph.analytics.pitch import draw_pitch


def main():
    print("Loading match data...")
    df = load_metrica_match(
        "data/raw/metrica/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "data/raw/metrica/Sample_Game_1_RawTrackingData_Away_Team.csv",
    )

    # Pick a mid-half frame where players are spread out
    target_frame = 10000
    team = "home"
    print(f"Building graphs for {team} team at frame {target_frame}...")

    # Build graphs using both rules
    G_del = build_frame_graph(df, frame=target_frame, team=team, edge_rule="delaunay")
    G_rad = build_frame_graph(df, frame=target_frame, team=team, edge_rule="radius", radius_m=15.0)

    # Extract node positions for nx.draw
    pos_del = nx.get_node_attributes(G_del, "pos")
    pos_rad = nx.get_node_attributes(G_rad, "pos")

    # Setup side-by-side plot
    print("Generating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.8))
    fig.patch.set_facecolor("#13261c")

    titles = ["Delaunay Triangulation", "Radius Graph (15m)"]
    graphs = [G_del, G_rad]
    positions = [pos_del, pos_rad]

    for i in range(2):
        ax = axes[i]
        draw_pitch(ax)
        ax.set_title(titles[i], color="white", fontsize=16, pad=15)

        G = graphs[i]
        pos = positions[i]

        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#fca311", width=2.0, alpha=0.8)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color="#ef4444", edgecolors="white", node_size=200
        )

        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white", font_weight="bold")

    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "graph_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Successfully saved to {out_path}")


if __name__ == "__main__":
    main()
