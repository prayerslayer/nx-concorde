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
_EDGE_WEIGHT_FORMAT = "FULL_MATRIX"
_EDGE_WEIGHT_DUMMY = 10_000
_RANDOM_SEED = 42


def _tsp_solver_factory(edge_weights: List[List], dimension: int):
    with tempfile.NamedTemporaryFile() as file:
        tsplib95.models.StandardProblem(
            type="TSP",
            dimension=dimension,
            edge_weight_type=_EDGE_WEIGHT_TYPE,
            edge_weight_format=_EDGE_WEIGHT_FORMAT,
            edge_weights=edge_weights,
        ).save(file.name)
        return TSPSolver.from_tspfile(file.name)


# @functools.lru_cache
def solve(edge_weights: List[List]) -> List[int]:
    """
    Solves a TSP tour.

    Args:
        edge_weights (List[List[float]]): Full matrix of distances between nodes.
    """
    solver = _tsp_solver_factory(edge_weights, dimension=len(edge_weights))
    solution = solver.solve(random_seed=_RANDOM_SEED)
    if not (solution.found_tour and solution.success):
        raise RuntimeError("No optimal route found.")
    return list(solution.tour)
