# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: nx-concorde
#     language: python
#     name: nx-concorde
# ---

from random import sample, seed, choice

import networkx as nx

from nx_concorde.graph import calc_tour

graph = nx.generators.random_geometric_graph(20, 0.4, seed=42)

seed(42)
visit_nodes = sample(graph.nodes(), 7)
visit_nodes

pos = nx.layout.spring_layout(graph, seed=42)
node_color = [1 if node in visit_nodes else 0 for node in graph.nodes()]

nx.draw(
    graph,
    pos=pos,
    node_color=node_color,
    with_labels=True,
    cmap="cool"
)

tour = calc_tour(graph, start_node=0, end_node=1, visit_nodes=visit_nodes)

# +
labels = {}

for idx, node in enumerate(tour):
    labels[node] = labels.get(node, []) + [str(idx)]
    
labels = {node: ", ".join(idxs) for node, idxs in labels.items()}
# -

nx.draw(
    graph,
    pos=pos,
    node_color=node_color,
    labels=labels,
    with_labels=True,
    cmap="cool"
)

graph = nx.generators.random_geometric_graph(50, 0.2, seed=42)
graph = graph.subgraph(max(nx.connected_components(graph), key=len))
len(graph.nodes())

seed(42)
start_node = choice(list(graph.nodes().keys()))
end_node = choice(list(graph.nodes().keys()))
visit_nodes = list(sample(graph.nodes(), 5))

from profilehooks import profile

import numpy as np


# +
@profile(immediate=True)
def calc_path_matrix_1(graph):
    return np.array(
        [
            [nx.astar_path(graph, source, target) for target in graph.nodes()]
            for source in graph.nodes()
        ],
        dtype="object",
    )

matrix1 = calc_path_matrix_1(graph)    
# -

import functools

# +
nx.astar_path_cached = functools.lru_cache(maxsize=None)(nx.astar_path)

@profile(immediate=True)
def calc_path_matrix_2(graph):
    nx.astar_path_cached.cache_clear()    
    matrix = []
    for source in graph.nodes():
        row = []
        for target in graph.nodes():
            if source > target:
                source_, target_ = target, source
                reverse = True
            else:
                source_, target_ = source, target
                reverse = False
            path = nx.astar_path_cached(graph, source_, target_)
            if reverse:
                path = path[::-1]
            row.append(path)
        matrix.append(row)
    return np.array(matrix, dtype="object")

matrix2 = calc_path_matrix_2(graph)
# -

import math

# +
pos = nx.layout.spring_layout(graph, seed=42)

# eucledian
def heuristic(source, target):
    x0, y0 = pos[source]
    x1, y1 = pos[target]
    return math.sqrt(math.pow(x0 - x1, 2) + math.pow(y0 - y1, 2))

for (source, target), data in graph.edges().items():
    data["weight"] = heuristic(source, target)
# -

for (source, target), data in graph.edges().items():
    data["weight"] = heuristic(source, target)


# +
@profile(immediate=True)
def calc_path_matrix_3(graph, heuristic=None, weight="weight"):
    nx.astar_path_cached.cache_clear()    
    matrix = []
    for source in graph.nodes():
        row = []
        for target in graph.nodes():
            if source > target:
                source_, target_ = target, source
                reverse = True
            else:
                source_, target_ = source, target
                reverse = False
            path = nx.astar_path_cached(graph, source_, target_, heuristic=heuristic, weight=weight)
            if reverse:
                path = path[::-1]
            row.append(path)
        matrix.append(row)
    return np.array(matrix, dtype="object")

matrix3 = calc_path_matrix_3(graph, heuristic)
# -

graph = nx.generators.random_geometric_graph(200, 0.2, seed=42)
graph = graph.subgraph(max(nx.connected_components(graph), key=len))
len(graph.nodes())

# +
pos = nx.layout.spring_layout(graph, seed=42)

# eucledian
def eucledian(source, target):
    x0, y0 = pos[source]
    x1, y1 = pos[target]
    return math.sqrt(math.pow(x0 - x1, 2) + math.pow(y0 - y1, 2))

for (source, target), data in graph.edges().items():
    data["weight"] = heuristic(source, target)


# +
@profile(immediate=True)
def calc_path_matrix_4(graph, heuristic=None, weight="weight"):
    nx.astar_path_cached.cache_clear()    
    matrix = []
    known_paths = {}
    def _heuristic(source, target):
        if source > target:
            source, target = target, source
        if (source, target) in known_paths:
            return known_paths[(source, target)]
        return heuristic(source, target)
    for source in graph.nodes():
        row = []
        for target in graph.nodes():
            if source > target:
                source_, target_ = target, source
                reverse = True
            else:
                source_, target_ = source, target
                reverse = False
            path = nx.astar_path_cached(graph, source_, target_, heuristic=_heuristic, weight=weight)
            known_paths[(source_, target_)] = len(path)
            if reverse:
                path = path[::-1]
            row.append(path)
        matrix.append(row)
    return np.array(matrix, dtype="object"), known_paths

matrix4, known_paths = calc_path_matrix_4(graph, heuristic)
# -

seed(42)
graph = nx.grid_2d_graph(20, 20)
graph = graph.subgraph(sample(graph.nodes(), 250))
graph = graph.subgraph(max(nx.connected_components(graph), key=len))
len(graph)

nx.draw(graph, pos={node: node for node in graph.nodes()})

from scipy.spatial import distance

block_distance = functools.partial(distance.minkowski, p=1)

nx.astar_path_cached.cache_info()

from multiprocessing import Pool


def astar_path_factory(graph, heuristic=None, weight=None):
    def astar_path(node_pair):
        source, target = node_pair
        return nx.astar_path(source=source, target=target, G=graph, heuristic=heuristic, weight=weight)
    return astar_path


astar_path = astar_path_factory(graph, heuristic=block_distance)

node_pairs = [(source, target) for target in graph.nodes() for source in graph.nodes() if target > source]

from pathos.multiprocessing import ProcessingPool as Pool

graph.nodes()

astar_path(((0, 0), (0, 1)))

agents = 2
chunksize = 3

with Pool(nodes=agents) as pool:
    result = pool.map(astar_path, node_pairs)

len(result)

node_pairs[100]

result[100]
