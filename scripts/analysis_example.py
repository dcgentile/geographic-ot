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
c0, c1 = np.random.randint(num_nodes), np.random.randint(num_nodes)
mu_0 = utils.random_geographic_concentration(ia_adj)
mu_1 = utils.random_geographic_concentration(ia_adj)
measures = np.stack((mu_0, mu_1)).T
t = 15
shortest_path_metric = utils.compute_graph_metric(graph)
diffusion_metric = utils.form_diffusion_map(ia_markov, t)
cost = shortest_path_metric**2
epsilon = 0.1
steps = 10


def geodesic_test(measures, cost, epsilon, steps):
    geodesic = total.build_geodesic(measures, cost, epsilon, method="", steps=steps)
    recovered_coords = np.zeros((steps, 2))
    true_coords = np.zeros((steps, 2))
    for i in range(steps):
        coords = total.simplex_regression(measures, geodesic[:, i], cost, epsilon)
        true_coords[i, :] = [1 - i / steps, i / steps]
        print(coords)
        recovered_coords[i, :] = coords
    t_rel_err = np.absolute(np.ones(steps) - recovered_coords[:, 0] / true_coords[:, 0])
    print(t_rel_err)


def projection_test(measures, cost, epsilon, steps, adj_mat):
    mu = utils.random_geographic_concentration(adj_mat)
    mu_coords = total.simplex_regression(measures, mu, cost, epsilon)
    print(mu_coords)
    approximate_mu = total.barycenter(mu_coords, measures, None, cost, epsilon)
    fig, (ax0, ax1) = plt.subplots(1, 2)
    nx.draw(graph, positions, node_color=mu, ax=ax0)
    nx.draw(graph, positions, node_color=approximate_mu, ax=ax1)
    plt.show()


projection_test(measures, cost, epsilon, steps, ia_adj)
