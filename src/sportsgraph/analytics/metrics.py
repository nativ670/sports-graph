import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist


def compute_frame_metrics(G: nx.Graph) -> dict:
    """
    Compute graph metrics:
    n_nodes, n_edges, density, mean_degree, largest_component_frac, algebraic_connectivity.

    Handles 0/1-node graphs explicitly (density and algebraic_connectivity are undefined for n<2).
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "density": None,
            "mean_degree": None,
            "largest_component_frac": None,
            "algebraic_connectivity": None,
        }

    if n_nodes == 1:
        return {
            "n_nodes": 1,
            "n_edges": 0,
            "density": None,
            "mean_degree": 0.0,
            "largest_component_frac": 1.0,
            "algebraic_connectivity": None,
        }

    density = nx.density(G)
    mean_degree = (2.0 * n_edges) / n_nodes

    components = list(nx.connected_components(G))
    largest_cc_size = len(max(components, key=len))
    largest_component_frac = largest_cc_size / n_nodes

    if nx.is_connected(G):
        # fiedler_vector can fail on some numerical matrices, so we just use algebraic_connectivity
        # Use default method for reliability
        try:
            alg_conn = nx.algebraic_connectivity(G)
        except Exception:
            alg_conn = 0.0
    else:
        alg_conn = 0.0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "mean_degree": mean_degree,
        "largest_component_frac": largest_component_frac,
        "algebraic_connectivity": alg_conn,
    }


def compute_geometric_baseline(positions: list[tuple[float, float]]) -> dict:
    """Compute the mean pairwise distance between all players."""
    n = len(positions)
    if n < 2:
        return {"mean_pairwise_distance": None}

    dists = pdist(positions)
    return {"mean_pairwise_distance": np.mean(dists)}
