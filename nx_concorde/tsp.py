"""
This module provides functions to solve the Travelling Salesman (TSP) problem
for a supermarket scenario.
"""

import functools
import tempfile
from typing import List

import numpy as np
import tsplib95
from concorde.tsp import TSPSolver

_EDGE_WEIGHT_TYPE = "EXPLICIT"
_RANDOM_SEED = 42


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
        edge_weights (List[List[float]]): Full matrix of distances between nodes.
    """
    solver = _tsp_solver_factory(
        edge_weights, dimension=dimension, edge_weight_format=edge_weight_format
    )
    solution = solver.solve(random_seed=_RANDOM_SEED)
    if not (solution.found_tour and solution.success):
        raise RuntimeError("No optimal route found.")
    return list(solution.tour)


def reshape_for_tsp_solver(array: List):
    def largest_prime_factor(n):
        i = 2
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
        return n

    return np.array(array).reshape(largest_prime_factor(len(array)), -1).tolist()
