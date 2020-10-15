"""

"""

from functools import lru_cache
from typing import Dict, Hashable, List

import networkx as nx
import numpy as np

from .tsp import solve

_DUMMY_DISTANCE = 10_000


@lru_cache()
def _calc_path_matrix(graph: nx.Graph) -> np.array:
    return np.array(
        [
            [nx.astar_path(graph, source, target) for target in graph.nodes()]
            for source in graph.nodes()
        ],
        dtype="object",
    )


def _calc_distance_matrix(graph: nx.Graph) -> np.array:
    return np.vectorize(len)(_calc_path_matrix(graph))


def _extend_tour(tour: List[int], path_matrix: np.array) -> List[Hashable]:
    tour_extended = []
    for start, end in zip(tour[:-1], tour[1:]):
        path = path_matrix[start][end]
        tour_extended.extend(path[:-1])
    tour_extended.append(path[-1])
    return tour_extended


def _calc_dummy_distance_matrix(
    distance_matrix: np.array, connected_node_idxs: List[int]
) -> np.array:
    dummy_row = np.repeat(_DUMMY_DISTANCE, len(distance_matrix))
    dummy_row[connected_node_idxs] = 0
    dummy_row = dummy_row.reshape(1, -1)
    dummy_col = np.append(dummy_row, 0).reshape(-1, 1)

    return np.append(np.append(distance_matrix, dummy_row, axis=0), dummy_col, axis=1)


def _remove_dummy_node(tour):
    return [idx for idx in tour if idx != max(tour)]


def _calc_node_idx_map(graph: nx.Graph) -> Dict:
    return {node: idx for idx, node in enumerate(graph.nodes())}


def _reorder_tour(tour: List[int], start: int = None, end: int = None):

    idx_start = [idx for idx, val in enumerate(tour) if val == start][0]
    idx_end = [idx for idx, val in enumerate(tour) if val == end][0]
    l = len(tour)
    tour = 2 * tour
    if tour[idx_start + l - 1] == end:
        return tour[idx_start : idx_start + l]
    return tour[idx_end : idx_end + l][::-1]


def calc_tour(
    graph,
    start_node: Hashable,
    end_node: Hashable,
    visit_nodes: List[Hashable] = None,
) -> List[Hashable]:
    path_matrix = _calc_path_matrix(graph)
    distance_matrix = _calc_distance_matrix(graph)
    node_idx_map = _calc_node_idx_map(graph)
    visit_nodes = visit_nodes or graph.nodes()
    visit_nodes = sorted(set(list(visit_nodes) + [start_node, end_node]))
    node_idxs = [node_idx_map[node] for node in visit_nodes]
    path_matrix = path_matrix[node_idxs, :][:, node_idxs]
    distance_matrix = distance_matrix[node_idxs, :][:, node_idxs]
    start_idx = [idx for idx, node in enumerate(visit_nodes) if node == start_node][0]
    end_idx = [idx for idx, node in enumerate(visit_nodes) if node == end_node][0]
    distance_matrix = _calc_dummy_distance_matrix(distance_matrix, [start_idx, end_idx])
    tour = solve(distance_matrix)
    tour = _remove_dummy_node(tour)
    tour = _reorder_tour(tour, start_idx, end_idx)
    tour = _extend_tour(tour, path_matrix)
    return tour


def plot_graph(graph, **kwargs):
    kwargs["pos"] = kwargs.get("pos") or {
        key: (node["x"], node["y"]) for key, node in graph.nodes().items()
    }
    kwargs["with_labels"] = kwargs["with_labels"] if "with_labels" in kwargs else True
    kwargs["node_color"] = kwargs.get("node_color") or [
        0 if node["type"] == "ENTRANCE" else 0.5 if node["type"] == "ISLE" else 1
        for node in graph.nodes().values()
    ]
    nx.draw(graph, **kwargs)
