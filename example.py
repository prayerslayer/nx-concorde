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

# # nx-concorde example

from random import sample, seed, choice
import functools

import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance

from nx_concorde.graph import calc_path_matrix, calc_distance_matrix, calc_tour

# Creating a 2-dimensional, grid-like graph with some holes.
seed(42)
graph = nx.grid_2d_graph(10, 10)
graph = graph.subgraph(sample(graph.nodes(), 80))
graph = graph.subgraph(max(nx.connected_components(graph), key=len))

# +
# Looking at the graph.
fig = plt.figure(figsize=(10, 10))

nx.draw(
    graph, pos={node: node for node in graph.nodes()}, ax=fig.gca(), with_labels=True
)
# -

# For a grid graph, block distance is a good heuristic.
block_distance = functools.partial(distance.minkowski, p=1)

# Calculating the path matrix. You can specify the number of threads with the `nodes` argument.
path_matrix = calc_path_matrix(graph, block_distance, nodes=1)

# Calculating the distance matrix from the node matrix.
distance_matrix = calc_distance_matrix(graph, path_matrix)

# Let's select 20 random nodes we want to visit.
NUM_VISIT_NODES = 20
visit_nodes = sorted(sample(graph.nodes(), k=NUM_VISIT_NODES))

# Let's select a random start and end node.
start_node = choice(list(graph.nodes()))
end_node = choice(list(graph.nodes()))

# We are calculating the optimal path that starts at start_node, ends at end_node and visits all visit_nodes.
tour = calc_tour(graph, start_node, end_node, visit_nodes, path_matrix, distance_matrix)

# Some labels to understand the path better.
labels = {}
for idx, node in enumerate(tour):
    labels[node] = labels.get(node, []) + [str(idx)]
labels = {node: ",\n".join(idxs) for node, idxs in labels.items()}

# +
# Let's plot the optimal tour.
fig = plt.figure(figsize=(10, 10))

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
