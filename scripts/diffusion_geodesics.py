#!/usr/bin/env python3
import sys

sys.path.append("../src")
import total
import utils
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import geopandas
from libpysal import weights
from gerrychain import Graph

filename = "../IA_counties/IA_counties.shp"
ia_graph = Graph.from_file(filename)
ia_counties = geopandas.read_file(filename)
ia_adj = nx.adjacency_matrix(ia_graph).toarray()
centroids = np.column_stack((ia_counties.centroid.x, ia_counties.centroid.y))
queen = weights.Queen.from_dataframe(ia_counties)
graph = queen.to_networkx()
positions = dict(zip(graph.nodes, centroids))
num_nodes = len(graph.nodes)


def comparison_video(measures, cost, epsilon, steps, filename, metric: str, p=2):
    geodesic = total.build_geodesic(
        measures,
        cost,
        epsilon,
        method="",
        steps=steps,
    )
    fig, ax0 = plt.subplots()
    fig.set_size_inches(20, 20)

    def update(frame):
        size = 1000
        nx.draw(graph, positions, ax=ax0, node_size=size, node_color=geodesic[:, frame])
        ax0.set_title(f"W{p}: {metric}\n Timestep : {frame + 1}")

    ani = animation.FuncAnimation(fig=fig, func=update, frames=steps, interval=100)
    ani.save(filename, writer="ffmpeg")


def animate_diff_dist(
    graph, steps=250, n=74, size=300, filename="ia_diffusion_evolution.mp4"
):
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    ia_markov = utils.adj_mat_to_markov_chain(ia_adj)

    def update(frame):
        if frame % (int(steps / 10)) == 0:
            print(frame)
        diff_cost = utils.form_diffusion_map(ia_markov, frame * 10)
        squared_diff_cost = diff_cost**2
        nx.draw(graph, positions, ax=ax0, node_size=size, node_color=diff_cost[n])
        nx.draw(
            graph, positions, ax=ax1, node_size=size, node_color=squared_diff_cost[n]
        )

    ani = animation.FuncAnimation(fig=fig, func=update, frames=steps, interval=100)
    ani.save(filename, writer="ffmpeg")


t = 20
graph_metric_cost = utils.compute_graph_metric(graph)
ia_markov = utils.adj_mat_to_markov_chain(ia_adj)
diff_cost = utils.form_diffusion_map(ia_markov, t)
squared_diff_cost = diff_cost**2
c0, c1 = 77, 37
cost, filename, metric = (
    diff_cost**2,
    "diff_metric_experiment.mp4",
    "diff sqr",
)
mu_0 = utils.random_geographic_concentration(cost, center=c0)
mu_1 = utils.random_geographic_concentration(cost, center=c1)
measures = np.stack((mu_0, mu_1)).T
epsilon = 0.1
steps = 50
comparison_video(measures, cost, epsilon, steps, filename, metric)
