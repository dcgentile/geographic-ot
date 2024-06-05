#!/usr/bin/env python3
import total
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import geopandas
from libpysal import weights
from gerrychain import Graph


filename = "./IA_counties/IA_counties.shp"
ia_graph = Graph.from_file(filename)
ia_counties = geopandas.read_file(filename)
fig, ax = plt.subplots()
ia_counties.plot(linewidth=1, ax=ax, edgecolor="grey", facecolor="white")
centroids = np.column_stack((ia_counties.centroid.x, ia_counties.centroid.y))
queen = weights.Queen.from_dataframe(ia_counties)
graph = queen.to_networkx()
positions = dict(zip(graph.nodes, centroids))

center_a = 34
center_b = 9

ia_adj = nx.adjacency_matrix(ia_graph)
ia_markov = total.adj_mat_to_markov_chain(ia_adj)
mu_0 = total.random_geographic_concentration(ia_adj.toarray(), center=center_a)
mu_1 = total.random_geographic_concentration(ia_adj.toarray(), center=center_b)
measures = np.stack((mu_0, mu_1)).T
epsilon = 0.1
for t in range(0, 4):
    time = 2**t
    ia_diffusion = total.form_diffusion_map(ia_markov, time)
    filename = f"./tufts-anims/ia_diffusion-time-{time}.mp4"
    steps = 50
    geodesic = total.build_geodesic(
        measures, ia_diffusion, epsilon, method="total", steps=steps
    )

    def update(frame):
        # steady_state_dist = find_steady_state(ia_markov)
        nx.draw(graph, positions, ax=ax, node_size=50, node_color=geodesic[:, frame])
        ax.set_title(
            f"Walking Along a Wasserstein Geodesic on a Discrete Space\n Timestep : {frame + 1}"
        )
        return ax

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=steps,
        interval=100,
    )
    ani.save(filename, writer="ffmpeg")

# fix an epsilon
# generate two random measures
# generate a random point along the geodesic between them
# use the optimizer to approximate the coordinates of the geodesic
# compute the geodesic at the approximate coordinates
# look at relative error between the approximate geodesic and the originally computed one
# do this a 1000 times per epsilon
# let epsilon go to 0
#
plt.clf()


def do_experiment(epsilon):
    num_nodes, _ = ia_adj.shape
    mu_0 = total.random_measure(num_nodes)
    mu_1 = total.random_measure(num_nodes)
    t = np.random.rand()
    measures = np.stack((mu_0, mu_1)).T
    bar = total.barycenter([t, 1 - t], measures, None, ia_diffusion, epsilon)
    recoverd_coords = total.simplex_regression(measures, bar, ia_diffusion, epsilon)
    approximate_bar = total.barycenter(
        recoverd_coords, measures, None, ia_diffusion, epsilon
    )

    return np.linalg.norm((bar - approximate_bar) / bar)


# epsilon_range = 0.01 * np.arange(100, 1, -1)
# bootstrap_cnt = 1000
#
# mean_error = []
#
# for epsilon in epsilon_range:
# experiments = [do_experiment(epsilon) for _ in range(bootstrap_cnt)]
# mean_error.append(np.average(experiments))
#
# fig, ax = plt.subplots()
# sns.lineplot(x=epsilon_range, y=mean_error, ax=ax)
# ax.set_title(
# "Mean Relative Error as a Function\n of the Regularization Parameter Epsilon"
# )
# ax.set_ylabel("Rel. Error")
# ax.set_xlabel(r"$\varepsilon$")
# plt.savefig("error.pdf")
#
