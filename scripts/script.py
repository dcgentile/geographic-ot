#!/usr/bin/env python3

import ot
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import geopandas
from libpysal import weights
from gerrychain import Graph
from math import sqrt
from numpy.linalg import inv, matrix_power
from scipy.optimize import minimize

graph = Graph.from_file("./IA_counties/IA_counties.shp")

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


def pairwise_diffusion_distance(mchain, steady_state, i, j):
    vec = ((mchain[i, :] - mchain[j, :]) ** 2) / steady_state

    return sqrt(np.sum(vec))


"""
given a Markov chain mchain and a time t, produce the matrix encoding the diffusion
distances between all pairs of nodes
"""


def compute_diffusion_map(mchain, t=1):
    time_scaled_chain = matrix_power(mchain, t)
    steady_state = find_markov_steady_state(mchain)
    dim = mchain.shape[0]
    D = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i + 1, dim):
            D[i, j] = pairwise_diffusion_distance(time_scaled_chain, steady_state, i, j)
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


def uniform_measure(num_nodes):
    weight = 1 / num_nodes
    v = [weight] * num_nodes
    return np.array(v)


def regularize_cost(matrix, epsilon=1.0):
    rows, columns = matrix.shape
    new_matrix = np.empty((rows, columns))
    for i in range(rows):
        for j in range(columns):
            ij_dist = matrix[i, j]
            new_matrix[i, j] = np.e ** (-1 * ij_dist / epsilon)
    return new_matrix


"""
given a pair of measures and cost matrix, build a matrix
whose columns are points along the geodesic path from the first measure to the second
"""


# def build_geodesic(measures, cost, epsilon=1.0, method='byhand', steps=10):
# assert(measures.shape[1] == 2)
# node_count = measures.shape[0]
# geodesic_steps = steps
# barycenters = np.empty((node_count,geodesic_steps))
# if method == "byhand" :
# for i in range(0, geodesic_steps):
# coordinates = np.array([i / geodesic_steps, 1 - i/geodesic_steps])
# barycenters[:,i] = barycenter(measures, cost, epsilon, coordinates, steps=10)
# else:
# print('computing using Python OT library')
# for i in range(0, geodesic_steps):
# coordinates = np.array([i / geodesic_steps, 1 - i/geodesic_steps])
# barycenters[:,i] = ot.barycenter(measures, cost, epsilon, weights=coordinates)
# return concatenate_measures(measures, barycenters)
def sinkhorn_differentiate(coords, measures, target, cost, iters=15, epsilon=0.1):
    num_nodes, num_measures = measures.shape
    b = np.empty((num_nodes, num_measures, iters))
    b[:, :, 0] = np.ones((num_nodes, num_measures))
    w = np.zeros(num_measures)
    r = np.zeros((num_nodes, num_measures))
    phi = np.empty((num_nodes, num_measures, iters))
    k = regularize_cost(cost, epsilon)
    for l in range(1, iters):
        b_l = np.empty((num_nodes, num_measures))
        for s in range(num_measures):
            m = measures[:, s]
            phi[:, s, l] = k.T @ (m / (k @ b[:, s, l - 1]))
        p = np.ones(num_nodes)
        for s in range(num_measures):
            p = p * (phi[:, s, l] ** coords[s])
        for i in range(num_measures):
            phi_col = phi[:, i, l]
            b_l[:, i] = np.divide(p, phi_col, out=np.zeros_like(p), where=phi_col != 0)
        b[:, :, l] = b_l
    # g = np.log(p / target) * p
    g = (p - target) * p
    for l in range(iters - 1, 0, -1):
        new_g = np.zeros(num_nodes)
        for m in range(num_measures):
            w[m] = w[m] + np.dot(np.log(phi[:, m, l]), g)
            r[:, m] = (
                -k.T
                @ (
                    k
                    @ ((coords[m] * g - r[:, m]) / phi[:, m, l])
                    * (measures[:, m] / ((k @ b[:, m, l - 1]) ** 2))
                )
                * b[:, m, l - 1]
            )
            new_g = new_g + r[:, m]
    return p, w


def barycenter(coords, measures, target, cost, epsilon=0.1, iters=15):
    p, _ = sinkhorn_differentiate(
        coords, measures, target, cost, epsilon=epsilon, iters=iters
    )
    return p


def logarithmic_change_of_variable(coords):
    v = np.exp(coords)
    n = np.linalg.norm(v, 1)
    return v / n


def rescaled_sinkhorn_differentiate(
    coords, measures, cost, target, iters=15, epsilon=0.1
):
    rescaled_coords = logarithmic_change_of_variable(coords)
    return sinkhorn_differentiate(
        rescaled_coords, measures, cost, target, iters=iters, epsilon=epsilon
    )


def squared_euc_loss(p, q):
    return 0.5 * (np.linalg.norm(p - q) ** 2)


def barycentric_loss(coordinates, measures, target, cost):
    bar = barycenter(
        logarithmic_change_of_variable(coordinates), measures, target, cost
    )
    return squared_euc_loss(bar, target)


def loss_gradient(coords, measures, cost, target):
    _, loss = sinkhorn_differentiate(coords, measures, cost, target)[1]
    return loss


ia_markov = adj_mat_to_markov_chain(nx.adjacency_matrix(graph))
num_nodes = ia_markov.shape[0]
num_measures = 2
t = 1
epsilon = 0.1
num_iters = 15
ia_diffusion = compute_diffusion_map(ia_markov, t)
measures = generate_measures(num_nodes, num_measures)
target = uniform_measure(num_nodes)
coords = generate_measures(2, 1)
bar = barycenter(
    coords, measures, target, ia_diffusion, epsilon=epsilon, iters=num_iters
)
initial_guess = uniform_measure(num_measures)
args = (measures, target, ia_diffusion)
minimize(
    barycentric_loss, initial_guess, method="L-BFGS-B", args=args, jac=loss_gradient
)
