#!/usr/bin/env python3

from math import sqrt
import networkx as nx
import numpy as np
from numpy.linalg import inv, matrix_power

"""
given a NetworkX graph, return the shortest distance metric
"""


def compute_graph_metric(graph):
    graph_dists = dict(nx.shortest_path_length(graph))
    sorted_dists = {k: sorted(v.items()) for k, v in graph_dists.items()}
    num_nodes = len(graph.nodes)
    dist_mat = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist_mat[i, j] = sorted_dists[i][j][1]

    return dist_mat


"""
given an an adjacency matrix for a graph G, produce the transition
matrix for the random walk on G
"""


def adj_mat_to_markov_chain(adj_mat):
    return inv(np.diag(np.sum(adj_mat, axis=1))) @ adj_mat


"""
given a valid cost matrix, return the indices which represent points
achieving the diameter of the set
"""


def max_dist(cost):
    return np.unravel_index(cost.argmax(), cost.shape)


"""
given a Markov chain p, produce the vector representing its steady state
"""


def find_markov_steady_state(p):
    dim = p.shape[0]
    q = p - np.eye(dim)
    ones = np.ones(dim)
    q = np.c_[q, ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)
    return np.linalg.solve(QTQ, bQT)


"""
given a Markov chain (or any power of it) mchain with steady state steady_state,
compute the diffusion distance between vertices i and j
"""


def diffusion_distance(mchain, steady_state, i, j):
    vec = ((mchain[i, :] - mchain[j, :]) ** 2) / steady_state

    return sqrt(np.sum(vec))


"""
given a Markov chain mchain and a time t, produce the matrix encoding the diffusion
distances between all pairs of nodes
"""


def form_diffusion_map(mchain, t):
    time_scaled_chain = matrix_power(mchain, t)
    steady_state = find_markov_steady_state(mchain)
    dim = mchain.shape[0]
    D = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i + 1, dim):
            D[i, j] = diffusion_distance(time_scaled_chain, steady_state, i, j)
            D[j, i] = D[i, j]

    return D


"""
returns a random probability measure on a graph with num_nodes number of nodes
"""


def random_measure(num_nodes):
    unnormalized_measure = np.random.rand(num_nodes)
    return unnormalized_measure / np.linalg.norm(unnormalized_measure, 1)


"""
given a number of nodes n and a number of measures to be produced m
return a n x m matrix whose columns correspond to probability measures
on a set of n elements
"""


def generate_measures(n: int, m: int):
    arr = np.empty((n, m))
    for i in range(m):
        arr[:, i] = random_measure(n)
    return arr


"""
given two measures head, tail and an array of measures measures, concatenate tail to the end of measures and measures to head
"""


def concatenate_measures(endpts, measures):
    return np.c_[np.c_[endpts[:, 0], measures], endpts[:, 1]]


"""
return a uniform distribution on num_nodes nodes
"""


def uniform_measure(num_nodes):
    weight = 1 / num_nodes
    v = [weight] * num_nodes
    return np.array(v)


"""
center is the index of a node in the graph
radius is the number of surrounding nodes on which the measure will be concentrated
the rest of the nodes have 1 / (num_nodes - (radius + 1)) weight on them
"""


def concentrated_measure(num_nodes, center, radius):
    measure = np.ones(num_nodes)
    measure = (1 / (num_nodes - (radius + 1))) * measure
    for i in range(center - radius, center + radius + 1, 1):
        measure[i] = 1 / (2 * radius + 1)

    return measure


def random_geographic_concentration(adj_mat, weight=10, center=None):
    num_nodes, _ = adj_mat.shape
    if center is None:
        center = np.random.randint(num_nodes)
    nbrhood = adj_mat[center]
    measure = np.ones(num_nodes)
    measure[center] *= weight
    for index, nbr in enumerate(nbrhood):
        if nbr == 1:
            measure[index] *= weight

    return measure / np.linalg.norm(measure, 1)
