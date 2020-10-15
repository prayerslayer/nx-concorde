"""

"""

from collections import OrderedDict
from copy import copy
from functools import lru_cache
from typing import Dict, Hashable, List

import networkx as nx
import numpy as np
from networkx.algorithms.shortest_paths.weighted import _weight_function
from pathos.multiprocessing import ProcessingPool as Pool

from .tsp import reshape_for_tsp_solver, solve

_DUMMY_COORD = None
_DUMMY_DISTANCE = 10_000


def _astar_path_factory(graph, heuristic=None, weight=None):
    def astar_path(node_pair):
        source, target = node_pair
        return nx.astar_path(
            source=source, target=target, G=graph, heuristic=heuristic, weight=weight
        )

    return astar_path


def calc_path_matrix(graph, heuristic=None, weight="weight", nodes=1):
    astar_path = _astar_path_factory(graph, heuristic=heuristic, weight=weight)
    node_pairs = [
        (source, target)
        for target in graph.nodes()
        for source in graph.nodes()
        if target > source
    ]
    with Pool(nodes=nodes) as pool:
        paths = pool.map(astar_path, node_pairs)
    return {frozenset(node_pair): path for node_pair, path in zip(node_pairs, paths)}


def _calc_distance(graph, path, weight="weight"):
    weight = _weight_function(graph, weight)
    return sum(weight(u, v, graph[u][v]) for u, v in zip(path[:-1], path[1:]))


def calc_distance_matrix(
    graph, path_matrix=None, heuristic=None, weight="weight", nodes=1
):
    path_matrix = path_matrix or calc_path_matrix(
        graph, heuristic=heuristic, weight=weight, nodes=1
    )
    return {
        node_pair: _calc_distance(graph, path, weight)
        for node_pair, path in path_matrix.items()
    }


def _extend_tour(tour: List[int], path_matrix: np.array) -> List[Hashable]:
    tour_extended = []
    for start, end in zip(tour[:-1], tour[1:]):
        path = path_matrix[start][end]
        tour_extended.extend(path[:-1])
    tour_extended.append(path[-1])
    return tour_extended


def _add_dummy_distance(distance_matrix, nodes, connected_nodes, inplace=False):
    dummy_node = (_DUMMY_COORD, _DUMMY_COORD)
    if not inplace:
        distance_matrix = copy(distance_matrix)
    for node in nodes:
        distance_matrix[frozenset((node, dummy_node))] = (
            0 if node in connected_nodes else _DUMMY_DISTANCE
        )
    return distance_matrix


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
    path_matrix: List = None,
    distance_matrix: List = None,
    **kwargs,
) -> List[Hashable]:
    path_matrix = path_matrix or calc_path_matrix(graph, **kwargs)
    distance_matrix = distance_matrix or calc_distance_matrix(
        graph, path_matrix=path_matrix
    )
    visit_nodes = list(visit_nodes or graph.nodes())
    visit_nodes = sorted(set(visit_nodes + [start_node, end_node]))
    visit_node_pairs = [
        frozenset((source, target))
        for target in visit_nodes
        for source in visit_nodes
        if target > source
    ]
    distance_matrix = {
        node_pair: distance_matrix[node_pair] for node_pair in visit_node_pairs
    }
    if start_node or end_node:
        distance_matrix = _add_dummy_distance(
            distance_matrix, visit_nodes, [start_node, end_node]
        )
        visit_nodes.append((None, None))
    return distance_matrix
    distance_matrix = [
        distance_matrix[frozenset((source, target))]
        for target in visit_nodes
        for source in visit_nodes
        if target > source
    ]
    return distance_matrix
    distance_matrix = reshape_for_tsp_solver(distance_matrix)
    return distance_matrix

    tour = solve(distance_matrix, len(visit_nodes))
    # tour = _remove_dummy_node(tour)
    # tour = _reorder_tour(
    #     tour, visit_nodes.index(start_node), visit_nodes.index(end_node)
    # )
    return tour
    # tour = _reorder_tour(tour, start_idx, end_idx)
    # tour = _extend_tour(tour, path_matrix)
    # return tour


# def plot_graph(graph, **kwargs):
#     kwargs["pos"] = kwargs.get("pos") or {
#         key: (node["x"], node["y"]) for key, node in graph.nodes().items()
#     }
#     kwargs["with_labels"] = kwargs["with_labels"] if "with_labels" in kwargs else True
#     kwargs["node_color"] = kwargs.get("node_color") or [
#         0 if node["type"] == "ENTRANCE" else 0.5 if node["type"] == "ISLE" else 1
#         for node in graph.nodes().values()
#     ]
#     nx.draw(graph, **kwargs)


# def largest_prime_factor(n):
#     i = 2
#     while i * i <= n:
#         if n % i:
#             i += 1
#         else:
#             n //= i
#     return n
