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
import functools

import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance

from nx_concorde.graph import calc_path_matrix, calc_distance_matrix, calc_tour
from nx_concorde.tsp import solve

seed(42)
graph = nx.grid_2d_graph(20, 20)
graph = graph.subgraph(sample(graph.nodes(), 250))
graph = graph.subgraph(max(nx.connected_components(graph), key=len))
len(graph)

# +
fig = plt.figure(figsize=(20, 20))

nx.draw(
    graph,
    pos={node: node for node in graph.nodes()},
    ax=fig.gca(),
    with_labels=True
)
# -

block_distance = functools.partial(distance.minkowski, p=1)

path_matrix = calc_path_matrix(graph, block_distance, nodes=None)

distance_matrix = calc_distance_matrix(graph, path_matrix)

NUM_VISIT_NODES = 20

visit_nodes = sorted(sample(graph.nodes(), k=NUM_VISIT_NODES))

start_node = choice(list(graph.nodes()))
end_node = choice(list(graph.nodes()))

print((start_node, end_node))

distance_matrix = calc_tour(graph, start_node, end_node, visit_nodes, path_matrix, distance_matrix)

visit_nodes = [(None, None)] + sorted(set(visit_nodes + [start_node, end_node]))

dm = [
    distance_matrix[frozenset((source, target))]
    for idx_target, target in enumerate(visit_nodes)
    for idx_source, source in enumerate(visit_nodes)
    if idx_source > idx_target
]

from nx_concorde.graph import reshape_for_tsp_solver

dm = reshape_for_tsp_solver(dm)

tour = solve(dm, len(visit_nodes), edge_weight_format="UPPER_ROW")

tour = [val for val in tour if val != 0]

tour

print((visit_nodes.index(start_node), (visit_nodes.index(end_node))))

from nx_concorde.graph import _reorder_tour

visit_nodes.index(start_node)

visit_nodes.index(end_node)

tour

tour = tour[::-1]

start_node

end_node

visit_nodes

tour

# +
path = []

for source, target in zip(tour[:-1], tour[1:]):
    sub_path = path_matrix[frozenset((visit_nodes[source], visit_nodes[target]))]
    
    if sub_path[0] != visit_nodes[source]:
        sub_path = sub_path[::-1]
    path.extend(sub_path[:-1])
path.append(sub_path[-1])

# +
labels = {}

for idx, node in enumerate(path):
    labels[node] = labels.get(node, []) + [str(idx)]
    
labels = {node: ",\n".join(idxs) for node, idxs in labels.items()}
# -

visit_nodes

# +
fig = plt.figure(figsize=(20, 20))

nx.draw(
    graph,
    pos={node: node for node in graph.nodes()},
    ax=fig.gca(),
    node_color=[1 if node in visit_nodes else 0 for node in graph.nodes()],
    labels=labels,
    with_labels=True,
    cmap="cool",
)
