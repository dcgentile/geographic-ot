#!/usr/bin/env python3

import numpy as np
import networkx as nx

"""
return the adjacency matrix of a graph on num_nodes vertices
"""


def cyclic_adjacency_matrix(num_nodes):
    m = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes - 1):
        m[i, i + 1] = 1
        m[i + 1, i] = 1
    m[0, num_nodes - 1] = 1
    m[num_nodes - 1, 0] = 1
    return m


"""
return the adjacency matrix of a complete graph on num_nodes vertices
"""


def complete_adjacency_matrix(num_nodes):
    m = np.ones((num_nodes, num_nodes))
    for i in range(num_nodes):
        m[i, i] = 0
    return m


"""
return the adjaceny matrix of a grid shape graph on num_nodes^2 vertices
"""


def grid_adjacency_matrix(num_nodes):
    m = np.zeros((num_nodes**2, num_nodes**2))
    for i in range(num_nodes**2 - 1):
        if (i + 1) % num_nodes > 0:
            m[i, i + 1] = 1
            m[i + 1, i] = 1
        if i + num_nodes < num_nodes**2:
            m[i, i + num_nodes] = 1
            m[i + num_nodes, i] = 1
    return m


# 0 - 1 - 2
# |   |   |
# 3 - 4 - 5
# |   |   |
# 6 - 7 - 8
