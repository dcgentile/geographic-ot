#!/usr/bin/env python3
import sys

sys.path.append("../src")
import ot
import total
import utils
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
import networkx as nx
import matplotlib.pyplot as plt
import geopandas
from libpysal import weights
from gerrychain import Graph

filename = "../IA_counties/IA_counties.shp"
voter_df = pd.read_csv("../data/voter_data.csv")
ia_graph = Graph.from_file(filename)
ia_counties = geopandas.read_file(filename)
ia_adj = nx.adjacency_matrix(ia_graph).toarray()
centroids = np.column_stack((ia_counties.centroid.x, ia_counties.centroid.y))
queen = weights.Queen.from_dataframe(ia_counties)
graph = queen.to_networkx()
positions = dict(zip(graph.nodes, centroids))

county_names = ia_counties["NAME10"]
march_df = voter_df[voter_df["Date"] == "March 2021"]
columns = ["County", "Democrat - Active"]
march_dem_df = march_df[columns]

sc = SpectralClustering(affinity="precomputed", n_init=1000)
sc.fit(ia_adj)
node_labels = {}
for node, label in enumerate(sc.labels_):
    if label in node_labels.keys():
        node_labels[label].append(node)
    else:
        node_labels[label] = [node]
num_nodes, _ = ia_adj.shape
num_refs = len(node_labels.keys())
refs = np.zeros((num_nodes, num_refs))
for i, label in zip(node_labels.keys(), range(num_refs)):
    ref = np.ones(num_nodes)
    for j in node_labels[label]:
        ref[j] = 100
    ref = ref / np.linalg.norm(ref, 1)
    refs[:, i] = ref


def data_by_index(data, col):
    colors = np.zeros(99)
    for index, county in enumerate(county_names):
        colors[index] = march_df[march_df["County"] == county][col]
    colors = colors / np.linalg.norm(colors, 1)
    return colors


dem_dist = data_by_index(march_df, "Democrat - Active")
rep_dist = data_by_index(march_df, "Republican - Active")
other_dist = data_by_index(march_df, "Other - Active")
np_dist = data_by_index(march_df, "No Party - Active")

epsilon = 0.01
cost = utils.compute_graph_metric(ia_graph) ** 2

random_simplex_point = utils.random_measure(num_refs)
# random_barycenter = total.barycenter(random_simplex_point, refs, None, cost, epsilon)
print(refs.shape)
random_barycenter = ot.barycenter(refs, cost, epsilon, weights=random_simplex_point)
recovered_coords = total.simplex_regression(refs, random_barycenter, cost, epsilon)
print(random_barycenter)
print(random_simplex_point)
print(recovered_coords)

# dem_coords = total.simplex_regression(refs, dem_dist, cost, epsilon)
# rep_coords = total.simplex_regression(refs, rep_dist, cost, epsilon)
# other_coords = total.simplex_regression(refs, other_dist, cost, epsilon)
# np_coords = total.simplex_regression(refs, np_dist, cost, epsilon)
#
#
# dem_reconstructed = total.barycenter(dem_coords, refs, None, cost, epsilon)
# print(dem_coords)
# print(dem_reconstructed)
# print(dem_dist)
#

# fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
# fig_inches = 15
# fig.set_size_inches(fig_inches, fig_inches)
# node_size = 50
# with_labels = False
# c0 = dem_reconstructed
# c1 = dem_dist
# c2 = refs[:, 2]
# ia_counties.plot(linewidth=1, ax=ax0, edgecolor="grey", facecolor="white")
# ia_counties.plot(linewidth=1, ax=ax1, edgecolor="grey", facecolor="white")
# ia_counties.plot(linewidth=1, ax=ax2, edgecolor="grey", facecolor="white")
# ia_counties.plot(linewidth=1, ax=ax3, edgecolor="grey", facecolor="white")
# nx.draw(
# graph,
# positions,
# ax=ax0,
# with_labels=with_labels,
# labels=county_names,
# node_size=node_size,
# node_color=c0,
# )
# nx.draw(
# graph,
# positions,
# ax=ax1,
# with_labels=with_labels,
# labels=county_names,
# node_size=node_size,
# node_color=c1,
# )
# nx.draw(
# graph,
# positions,
# ax=ax2,
# with_labels=with_labels,
# labels=county_names,
# node_size=node_size,
# node_color=c2,
# )
# nx.draw(
# graph,
# positions,
# ax=ax3,
# with_labels=with_labels,
# labels=county_names,
# node_size=node_size,
# node_color=sc.labels_,
# )
# plt.show()
#
