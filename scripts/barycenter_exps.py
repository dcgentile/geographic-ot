#!/usr/bin/env python3
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
import total

voter_df = pd.read_csv("./data/voter_data.csv")
voter_df.head()


def draw_graph(G, seed=0):
    pos = nx.spring_layout(G, seed=seed)
    nx.draw(G, pos=pos, node_size=500, with_labels=True)
    return fig


def draw_adjacency_matrix(adjacency_matrix, seed):
    fig, ax = plt.subplots()
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    return draw_graph(G, seed)


def draw_grid(G, ax, node_size=400, node_color="lightgreen"):
    pos = {(x, y): (y, -x) for x, y in G.nodes()}
    nx.draw(
        G,
        pos=pos,
        node_color=node_color,
        ax=ax,
        # with_labels=True,
        node_size=node_size,
    )
    return ax


def compute_graph_metric(G):
    graph = nx.convert_node_labels_to_integers(G)
    graph_dists = dict(nx.shortest_path_length(graph))
    sorted_dists = {k: sorted(v.items()) for k, v in graph_dists.items()}
    num_nodes = len(graph.nodes)
    dist_mat = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist_mat[i, j] = sorted_dists[i][j][1]

    return dist_mat


grid_n = 10
grid = nx.grid_graph((grid_n, grid_n))
cost = compute_graph_metric(grid) ** 2
c0, c1, c2 = 12, 88, 82
mu_0 = total.random_geographic_concentration(cost, center=c0)
mu_1 = total.random_geographic_concentration(cost, center=c1)
mu_2 = total.random_geographic_concentration(cost, center=c2)
reference_measures = np.stack((mu_0, mu_1, mu_2)).T
coords = np.array([1 / 3, 1 / 3, 1 / 3])
epsilon = 0.1
mu = total.barycenter(coords, reference_measures, None, cost, epsilon)
iters = 10
num_nodes = len(grid.nodes)
bars = np.zeros((num_nodes, iters))
bars[:, 0] = total.barycenter(coords, reference_measures, None, cost, epsilon)
for i in range(1, iters):
    new_refs = np.stack((bars[:, i - 1], mu_1, mu_2)).T
    new_mu = total.barycenter(coords, new_refs, None, cost, epsilon)
    bars[:, i] = new_mu


def animate(graph, bars, refs, filename, frame_size, node_size):
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.set_size_inches(frame_size * 2, frame_size)
    _, steps = bars.shape
    r0, r1, r2 = refs[:, 0], refs[:, 1], refs[:, 2]
    r_sum = r0 + r1 + r2

    def update(frame):
        print(frame)
        bar = bars[:, frame]
        draw_grid(graph, ax=ax0, node_color=r_sum, node_size=node_size)
        draw_grid(graph, ax=ax1, node_color=bar, node_size=node_size)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=steps, interval=1000)
    ani.save(filename, writer="ffmpeg")


animate(grid, bars, reference_measures, "iterated_barycenters.mp4", 5, 200)

from tqdm import tqdm

num_refs = 3
recovered_coordinates = np.zeros((num_refs, iters))
for i in tqdm(range(iters)):
    recovered_coordinates[:, i] = total.simplex_regression(
        reference_measures, bars[:, i], cost, epsilon
    ).T

fig, (ax0, ax1) = plt.subplots(1, 2)
fig.set_size_inches(10, 5)
draw_grid(grid, ax=ax0, node_color=(mu_0 + mu_1 + mu_2))
draw_grid(grid, ax=ax1, node_color=mu)
plt.savefig("mybar.pdf", bbox_inches="tight")

for i in range(100):
    if i % 10 == 0:
        print(recovered_coordinates[:, i])
