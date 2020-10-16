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

seed(42)
graph = nx.grid_2d_graph(20, 20)
graph = graph.subgraph(sample(graph.nodes(), 250))
graph = graph.subgraph(max(nx.connected_components(graph), key=len))
len(graph)

# +
fig = plt.figure(figsize=(20, 20))

nx.draw(
    graph, pos={node: node for node in graph.nodes()}, ax=fig.gca(), with_labels=True
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

tour = calc_tour(graph, start_node, end_node, visit_nodes, path_matrix, distance_matrix)

# +
labels = {}

for idx, node in enumerate(tour):
    labels[node] = labels.get(node, []) + [str(idx)]

labels = {node: ",\n".join(idxs) for node, idxs in labels.items()}

# +
fig = plt.figure(figsize=(20, 20))

nx.draw(
    graph,
    pos={node: node for node in graph.nodes()},
    ax=fig.gca(),
    node_color=[
        1 if node in visit_nodes + [start_node, end_node] else 0
        for node in graph.nodes()
    ],
    labels=labels,
    with_labels=True,
    cmap="cool",
)
# -
