#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import geopandas
from libpysal import weights
from gerrychain import Graph

filename = "./IA_counties/IA_counties.shp"
voter_df = pd.read_csv("./data/voter_data.csv")
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


def data_by_index(data, col):
    colors = np.zeros(99)
    for index, county in enumerate(county_names):
        colors[index] = march_df[march_df["County"] == county][col]
    colors = colors / np.linalg.norm(colors, 1)
    return colors


fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
fig_inches = 15
fig.set_size_inches(fig_inches, fig_inches)
node_size = 500
dem_color = data_by_index(march_df, "Democrat - Active")
rep_color = data_by_index(march_df, "Republican - Active")
other_color = data_by_index(march_df, "Other - Active")
np_color = data_by_index(march_df, "No Party - Active")
ia_counties.plot(linewidth=1, ax=ax0, edgecolor="grey", facecolor="white")
ia_counties.plot(linewidth=1, ax=ax1, edgecolor="grey", facecolor="white")
ia_counties.plot(linewidth=1, ax=ax2, edgecolor="grey", facecolor="white")
ia_counties.plot(linewidth=1, ax=ax3, edgecolor="grey", facecolor="white")
nx.draw(
    graph,
    positions,
    ax=ax0,
    with_labels=True,
    labels=county_names,
    node_size=node_size,
    node_color=dem_color,
)
nx.draw(
    graph,
    positions,
    ax=ax1,
    with_labels=True,
    labels=county_names,
    node_size=node_size,
    node_color=rep_color,
)
nx.draw(
    graph,
    positions,
    ax=ax2,
    with_labels=True,
    labels=county_names,
    node_size=node_size,
    node_color=other_color,
)
nx.draw(
    graph,
    positions,
    ax=ax3,
    with_labels=True,
    labels=county_names,
    node_size=node_size,
    node_color=np_color,
)
