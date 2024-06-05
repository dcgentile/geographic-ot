#!/usr/bin/env python3

import sys

sys.path.append("../src")
import utils
import total
import numpy as np
import networkx as nx
import geopandas
from libpysal import weights
from gerrychain import Graph
import matplotlib.pyplot as plt
import argparse
from sklearn.cluster import SpectralClustering

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="shape file of graph")
args = parser.parse_args()
ia_graph = Graph.from_file(args.filename)
ia_counties = geopandas.read_file(args.filename)
ia_adj = nx.adjacency_matrix(ia_graph).toarray()
centroids = np.column_stack((ia_counties.centroid.x, ia_counties.centroid.y))
queen = weights.Queen.from_dataframe(ia_counties)
graph = queen.to_networkx()
positions = dict(zip(graph.nodes, centroids))
num_nodes = len(graph.nodes)
ia_markov = utils.adj_mat_to_markov_chain(ia_adj)
shortest_path_metric = utils.compute_graph_metric(graph)
shortest_path_squared = shortest_path_metric**2
SpectralClustering = SpectralClustering(affinity="precomputed")
SpectralClustering.fit(ia_adj)
labels = SpectralClustering.labels_
num_clusters = len(np.unique(labels))
references = np.zeros((num_nodes, num_clusters))
# build a matrix with columns representing reference measures,
# with references corresponding to computed graph clusters
for node, label in enumerate(labels):
    references[node, label] = 1
# normalize the reference measures
for col in range(num_clusters):
    references[:, col] = references[:, col] / np.linalg.norm(references[:, col], 1)
epsilon = 0.01
# random_mu = utils.random_measure(num_nodes)
mu = references[:, 0]
coords = total.simplex_regression(references, mu, shortest_path_squared, epsilon)
print(coords)
approximate_mu = total.barycenter(
    coords, references, None, shortest_path_squared, epsilon
)
# fig, (ax0, ax1) = plt.subplots(1, 2)
# nx.draw(graph, positions, node_color=random_mu, ax=ax0)
# nx.draw(graph, positions, node_color=approximate_mu, ax=ax1)
# plt.show()
#
