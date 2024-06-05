#!/usr/bin/env python3

import total
import ot
import numpy as np
from numpy import average
from numpy.linalg import norm
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

import geopandas
from libpysal import weights
from gerrychain import Graph

"""
a script to gather data about how the diffusion timescale
affects Wasserstein barycenters on a graph
"""


def main():
    graph = Graph.from_file("./IA_counties/IA_counties.shp")
    ia_markov = total.adj_mat_to_markov_chain(nx.adjacency_matrix(graph))
    num_nodes = ia_markov.shape[0]
    t = 10
    num_measures = 2
    epsilon = 0.1
    steps = 50
    cost = total.form_diffusion_map(ia_markov, t) ** 2
    barycenter_discrepancy_experiment(
        cost, num_measures, num_nodes, epsilon, steps, img=True
    )
    plt.savefig("sqr-cost-tot-pot-comp.pdf")
    regression_filename = "./img/regression-relarive-err-finegrain.png"
    reg_graph(cost, num_nodes, regression_filename)
    # gif_name = "./img/diffusion_evo.gif"
    # make_gif(ia_markov, 10, gif_name)


#


def draw_example(cost, num_nodes, num_measures, epsilon, steps):
    example_geo = barycenter_discrepancy_experiment(
        cost,
        num_measures=2,
        num_nodes=num_nodes,
        epsilon=epsilon,
        steps=steps,
        img=True,
    )
    plt.savefig("./img/example-geodesic.png")


def acc_graph(cost, num_nodes, num_measures, steps):
    delta_max = 100
    epsilon_run = [delta / delta_max for delta in range(1, delta_max + 1)]
    average_diffs = []
    for delta in range(1, delta_max + 1):
        epsilon = delta / delta_max
        boots = 1000
        tot_v_pot_diffs = [
            barycenter_discrepancy_experiment(
                cost,
                num_measures=num_measures,
                num_nodes=num_nodes,
                epsilon=epsilon,
                steps=steps,
            )
            for _ in range(boots)
        ]
        average_diffs.append(average(tot_v_pot_diffs))
    sns.lineplot(x=epsilon_run, y=average_diffs).set(yscale="log")
    plt.title("Mean Squared Error between POT and TOTAL Barycenters")
    plt.savefig("./img/total-vs-pot-accuracy.png", dpi=600)


def reg_graph(cost, num_nodes, filename, max_measures=10, delta_max=50):
    error_mat = np.empty((max_measures, delta_max))
    epsilon_run = [delta / delta_max for delta in range(1, delta_max + 1)]
    for n in tqdm(range(2, 2 + max_measures)):
        num_measures = n
        for delta in tqdm(range(1, delta_max + 1)):
            epsilon = delta / delta_max
            boots = 100
            recovery_diffs = [
                regression_experiment(
                    cost,
                    num_measures=num_measures,
                    num_nodes=num_nodes,
                    epsilon=epsilon,
                )
                for _ in range(boots)
            ]
            error_mat[n - 2, delta - 1] = average(recovery_diffs)
    ax1 = sns.heatmap(error_mat)
    ax1.set_xticks(np.arange(1, delta_max + 1))
    ax1.set_xticklabels(f"{c:.1f}" for c in epsilon_run)
    plt.title("Relative Error between True and Approximated Barycentric Coordinates")
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel("Number of Measures")
    plt.savefig(filename, dpi=600)


def make_gif(mchain, steps, filename, epsilon=0.01):
    filepath = "./IA_counties/IA_counties.shp"
    ia_counties = geopandas.read_file(filepath)
    fig, _ = plt.subplots()
    centroids = np.column_stack((ia_counties.centroid.x, ia_counties.centroid.y))
    queen = weights.Queen.from_dataframe(ia_counties)
    graph = queen.to_networkx()
    positions = dict(zip(graph.nodes, centroids))
    ax = ia_counties.plot(linewidth=1, edgecolor="grey", facecolor="white")
    num_nodes = mchain.shape[0]
    num_measures = 3
    concentrated_measures = np.empty((num_nodes, num_measures))
    for i in range(num_measures):
        concentrated_measures[:, i] = total.concentrated_measure(
            num_nodes, np.random.randint(num_nodes), 2
        )
    coords = total.uniform_measure(num_measures)
    fig.set_size_inches(16, 8)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ia_counties.plot(linewidth=1, ax=ax1, edgecolor="grey", facecolor="white")
    nx.draw(
        graph, positions, ax=ax1, node_size=50, node_color=concentrated_measures[:, 0]
    )
    ia_counties.plot(linewidth=1, ax=ax2, edgecolor="grey", facecolor="white")
    nx.draw(
        graph, positions, ax=ax2, node_size=50, node_color=concentrated_measures[:, 1]
    )
    ia_counties.plot(linewidth=1, ax=ax3, edgecolor="grey", facecolor="white")
    nx.draw(
        graph, positions, ax=ax3, node_size=50, node_color=concentrated_measures[:, 2]
    )

    def update(frame):
        diff_map = total.form_diffusion_map(mchain, 10 * frame)
        ia_counties.plot(linewidth=1, ax=ax4, edgecolor="grey", facecolor="white")
        bar = total.barycenter(
            coords, concentrated_measures, None, diff_map, epsilon=epsilon
        )
        nx.draw(graph, positions, ax=ax4, node_size=50, node_color=10 * bar)
        ax4.set_title(
            f"Uniform Barycenter of Measures \n at Diffusion Time: {100 * frame}"
        )

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=steps, interval=500, repeat=True
    )
    ani.save(filename=filename, writer="imagemagick")


"""
do a single experiment:
input: cost matrix C, number of measures s, regularization parameter epsilon
1. generate s random measures on the vertex set
2. compute the barycenter of the measures with cost C and regularization parameter epsilon
   using the TOTAL implementation
3. compute the barycenter again using the POT implementation
4. return the L^2 difference of the barycenters
"""


def barycenter_discrepancy_experiment(
    cost, num_measures, num_nodes, epsilon, steps, img=False
):
    measures = total.generate_measures(num_nodes, num_measures)
    if img:
        tot_geo = total.build_geodesic(
            measures, cost, epsilon=epsilon, method="total", steps=steps
        )
        pot_geo = total.build_geodesic(
            measures, cost, epsilon=epsilon, method="pot", steps=steps
        )
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        for ax in [ax1, ax2, ax3]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.axis("off")
        fig.set_size_inches((20, 10))
        ax1.set_title("Geodesic via POT Implementation")
        ax2.set_title("Geodesic via TOTAL Implementation")
        ax3.set_title("Relative Error Between Results")
        ax1.set_xlabel("Step Along Geodesic")
        ax2.set_xlabel("Step Along Geodesic")
        ax1.set_ylabel("Node Label")
        ax2.set_ylabel("Node Label")
        sns.heatmap(
            pot_geo,
            # cmap=sns.color_palette("icefire", as_cmap=True),
            ax=ax1,
            xticklabels=10,
            yticklabels=10,
        )
        sns.heatmap(
            tot_geo,
            # cmap=sns.color_palette("icefire", as_cmap=True),
            ax=ax2,
            xticklabels=10,
            yticklabels=10,
        )
        sns.heatmap(
            abs(tot_geo - pot_geo) / pot_geo, ax=ax3, xticklabels=10, yticklabels=10
        )
        return fig
    else:
        coords = total.generate_measures(num_measures, 1)
        tot_bar = total.barycenter(coords, measures, None, cost, epsilon)
        pot_bar = ot.barycenter(measures, cost, epsilon, weights=coords[:, 0])
        return norm(tot_bar - pot_bar) ** 2


"""
do a single experiment:
input: cost matrix C, number of measures s, regularization parameter epsilon
1. generate s random measures on the vertex set
2. compute the barycenter of the measures with cost C and regularization parameter epsilon
   using the TOTAL implementation
3. compute the barycentric coordinates TOTAL implementation
4. return the relative error of the approximate barycentric coordinates
"""


def regression_experiment(cost, num_measures, num_nodes, epsilon):
    measures = total.generate_measures(num_nodes, num_measures)
    coords = total.generate_measures(num_measures, 1)
    print(coords.shape, measures.shape)
    bar = total.barycenter(coords, measures, None, cost, epsilon=epsilon)
    recovered_coords = total.simplex_regression(measures, bar, cost, epsilon)
    return abs(1 - coords.T / recovered_coords)


if __name__ == "__main__":
    main()
