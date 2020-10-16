"""
This module provides functions to solve the Travelling Salesman (TSP) problem.
"""
import tempfile
from typing import List

import numpy as np
import tsplib95
from concorde.tsp import TSPSolver

_EDGE_WEIGHT_TYPE = "EXPLICIT"


def _tsp_solver_factory(edge_weights: List, dimension: int, edge_weight_format: str):
    with tempfile.NamedTemporaryFile() as file:
        tsplib95.models.StandardProblem(
            type="TSP",
            dimension=dimension,
            edge_weight_type=_EDGE_WEIGHT_TYPE,
            edge_weight_format=edge_weight_format,
            edge_weights=edge_weights,
        ).save(file.name)
        return TSPSolver.from_tspfile(file.name)


# @functools.lru_cache
def solve(
    edge_weights: List, dimension: int, edge_weight_format: str = "UPPER_ROW"
) -> List[int]:
    """
    Solves a TSP tour.

    Args:
        edge_weights (List[List[float]]): Explicit edge weighs.
        edge_weight_format (str): Format of the edge weights.

    Returns:
        List[int]: Tour of node indexes.
    """
    solver = _tsp_solver_factory(
        edge_weights, dimension=dimension, edge_weight_format=edge_weight_format
    )
    solution = solver.solve()
    if not (solution.found_tour and solution.success):
        raise RuntimeError("No optimal route found.")
    return list(solution.tour)


def reshape_for_tsp_solver(edge_weights: List) -> List[List]:
    """
    Reshapes a flat list of edge weights into a 2-dimensional matrix
    that can the TSP solver can understand.
    """

    def largest_prime_factor(num):
        i = 2
        while i * i <= num:
            if num % i:
                i += 1
            else:
                num //= i
        return num

    return (
        np.array(edge_weights)
        .reshape(largest_prime_factor(len(edge_weights)), -1)
        .tolist()
    )
