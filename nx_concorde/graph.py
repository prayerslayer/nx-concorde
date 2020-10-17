"""
This module contains functions to calculate TSP tours for networkx graphs.
"""

from copy import copy
from typing import Callable, Dict, Hashable, List

import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from pathos.multiprocessing import ProcessingPool as Pool

from .tsp import reshape_for_tsp_solver, solve

_DUMMY_NODE = (None, None)
_DUMMY_DISTANCE = 10_000


def _astar_path_factory(graph, heuristic=None, weight=None):
    def astar_path(node_pair):
        source, target = node_pair
        return nx.astar_path(
            source=source, target=target, G=graph, heuristic=heuristic, weight=weight
        )

    return astar_path


def calc_path_matrix(
    graph: nx.Graph, heuristic: Callable = None, weight: str = "weight", nodes: int = 1
) -> Dict[Hashable, List[Hashable]]:
    """
    Calculates a shortest path matrix between all combinations of nodes in the graph.
    """
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


def _calc_distance(graph, path, weight="weight") -> float:
    weight = _weight_function(graph, weight)
    return sum(weight(u, v, graph[u][v]) for u, v in zip(path[:-1], path[1:]))


def calc_distance_matrix(
    graph, path_matrix=None, heuristic=None, weight="weight", nodes=1
) -> Dict[Hashable, float]:
    """
    Calculates the shortest path length matrix between all combinations of nodes in the graph.
    """
    path_matrix = path_matrix or calc_path_matrix(
        graph, heuristic=heuristic, weight=weight, nodes=nodes
    )
    return {
        node_pair: _calc_distance(graph, path, weight)
        for node_pair, path in path_matrix.items()
    }


def _extend_tour(
    tour: List[Hashable], path_matrix: Dict[Hashable, List[Hashable]]
) -> List[Hashable]:
    path = []

    for source, target in zip(tour[:-1], tour[1:]):
        sub_path = path_matrix[frozenset((source, target))]

        if sub_path[0] != source:
            sub_path = sub_path[::-1]
        path.extend(sub_path[:-1])
    path.append(sub_path[-1])
    return path


def _add_dummy_distance(distance_matrix, nodes, connected_nodes, in_place=False):
    if not in_place:
        distance_matrix = copy(distance_matrix)
    for node in nodes:
        distance_matrix[frozenset((node, _DUMMY_NODE))] = (
            0 if node in connected_nodes else _DUMMY_DISTANCE
        )
    return distance_matrix


def _reorder_tour(tour: List[Hashable], start: Hashable = None, end: Hashable = None):

    start_idx = tour.index(start)
    end_idx = tour.index(end)
    tour_length = len(tour)
    tour = 2 * tour
    if tour[start_idx + tour_length - 1] == end:
        return tour[start_idx : start_idx + tour_length]
    return tour[end_idx : end_idx + tour_length][::-1]


def _to_upper_row(
    distance_matrix: Dict[Hashable, float], nodes: List[Hashable]
) -> List[float]:
    return [
        distance_matrix[frozenset((source, target))]
        for idx_target, target in enumerate(nodes)
        for idx_source, source in enumerate(nodes)
        if idx_source > idx_target
    ]


# pylint: disable=too-many-arguments
def calc_tour(
    graph,
    start_node: Hashable,
    end_node: Hashable,
    visit_nodes: List[Hashable] = None,
    path_matrix: Dict[Hashable, List[Hashable]] = None,
    distance_matrix: Dict[Hashable, float] = None,
) -> List[Hashable]:
    """
    Calculates the TSP tour for graph that starts at start_node,
    ends at end_node and passes all visit_nodes.
    """
    path_matrix = path_matrix or calc_path_matrix(graph)
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
    dummy = start_node or end_node
    if dummy:
        distance_matrix = _add_dummy_distance(
            distance_matrix, visit_nodes, [start_node, end_node]
        )
        visit_nodes = [_DUMMY_NODE] + visit_nodes
    distance_matrix = _to_upper_row(distance_matrix, visit_nodes)
    distance_matrix = reshape_for_tsp_solver(distance_matrix)
    tour = solve(distance_matrix, len(visit_nodes))
    tour = [visit_nodes[idx] for idx in tour]
    if dummy:
        tour = [node for node in tour if node != _DUMMY_NODE]
        tour = _reorder_tour(tour, start_node, end_node)
    tour = _extend_tour(tour, path_matrix)
    return tour
