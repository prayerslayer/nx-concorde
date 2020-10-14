"""
This module contains utilities.
"""
from typing import Dict, List, Tuple

import networkx as nx

_DIRECTIONS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}

_MAZE_TYPES = {
    1: "ISLE",
    2: "ENTRANCE",
    3: "EXIT",
}


def read_maze(filepath: str) -> List[List[int]]:
    with open(filepath, mode="r") as file_pointer:
        maze = file_pointer.read()
    return [[int(val) for val in row] for row in maze.split("\n")[::-1] if row]


def maze_to_graph(maze: List[List[int]]) -> nx.Graph:
    graph = nx.Graph()

    graph.add_nodes_from(
        ((x, y), {"type": _MAZE_TYPES[val], "x": x, "y": y})
        for y, row in enumerate(maze)
        for x, val in enumerate(row)
        if val
    )
    graph.add_edges_from(
        ((x, y), (x + dx, y + dy))
        for x, y in graph.nodes()
        for (dx, dy) in _DIRECTIONS.values()
        if graph.has_node((x + dx, y + dy))
    )

    return graph
