#!/usr/bin/python3.10
# coding=utf-8

# author: Veronika Nevarilova (cnevar00)

import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# muzete pridat vlastni knihovny


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Converts the dataframe to  geopandas.GeoDataFrame
        with correct encoding"""

    # remove null values
    df = df.dropna(subset=['d', 'e'])

    # create geodataframe with geometry from 'd' and 'e' columns
    gdf = geopandas.GeoDataFrame(
        df,
        geometry=geopandas.points_from_xy(df['d'], df['e'], crs='EPSG:5514')
    )
    return gdf


def setlims(ax: np.ndarray):
    """ Sets the limits of graphs to the greater values """
    x_lim1 = ax[0].get_xlim()
    y_lim1 = ax[0].get_ylim()

    x_lim2 = ax[1].get_xlim()
    y_lim2 = ax[1].get_ylim()

    # getting the size of the limit for x axis
    new_x_lim = [min(x_lim1[0], x_lim2[0]), max(x_lim1[1], x_lim2[1])]

    # getting the size of the limit for y axis
    new_y_lim = [min(y_lim1[0], y_lim2[0]), max(y_lim1[1], y_lim2[1])]

    # set both subplots the same
    ax[0].set_xlim(new_x_lim)
    ax[0].set_ylim(new_y_lim)

    ax[1].set_xlim(new_x_lim)
    ax[1].set_ylim(new_y_lim)

    # remove showing the axes
    ax[0].axis('off')
    ax[1].axis('off')


def clip_invalid_points(gdf: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    bounds = [-700000.0, -1300000.0, -400000.0, -900000.0]
    gdf = gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
    return gdf


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Plots a graph with accidents  """

    working_gdf = gdf.copy()

    # data from JHM region and only accidents with animals
    working_gdf = working_gdf[(working_gdf['region'] == 'JHM')
                              & (working_gdf['p10'] == 4)]
    working_gdf = clip_invalid_points(working_gdf)

    working_gdf['p2a'] = pd.to_datetime(working_gdf['p2a'])

    gdf_2021 = working_gdf[working_gdf['p2a'].dt.year == 2021]
    gdf_2022 = working_gdf[working_gdf['p2a'].dt.year == 2022]

    # set Web Mercator projection
    gdf_2021 = gdf_2021.to_crs(epsg=3857)
    gdf_2022 = gdf_2022.to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharex=False, sharey=False)

    gdf_2021.plot(ax=ax[0], markersize=3, color='red')

    ax[0].set_title('JHM kraj (2021)')

    # add map from OpenStreetMap
    ctx.add_basemap(ax=ax[0], crs=gdf_2021.crs.to_string(),
                    source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.9)

    gdf_2022.plot(ax=ax[1], markersize=3, label="Nehody 2022", color='red')
    ctx.add_basemap(ax=ax[1], crs=gdf_2022.crs.to_string(),
                    source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.9)

    # visual settings
    ax[1].set_title('JHM kraj (2022)')

    # set appropriate limits
    setlims(ax)

    plt.tight_layout()

    if fig_location:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Plotting a graph with location of all acccidents in region
        grouped into clusters """

    working_gdf = gdf.copy()

    # accidents with significant alcohol rate
    working_gdf = working_gdf[(working_gdf['region'] == 'JHM')
                              & (working_gdf['p11'] >= 4)]
    working_gdf = clip_invalid_points(working_gdf)

    # set Web Mercator projection
    working_gdf = working_gdf.to_crs(epsg=3857)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    working_gdf.plot(ax=ax, markersize=2, label="Nehody 2021", color='red')
    ctx.add_basemap(ax=ax, crs=working_gdf.crs.to_string(),
                    source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.9)
    ax.axis('off')
    ax.set_title('Nehody v JHM kraji s významnou měrou alkoholu')

    # making clusters
    coordinates = working_gdf[['geometry']].copy()
    coordinates['x'] = working_gdf.geometry.x
    coordinates['y'] = working_gdf.geometry.y
    coords = coordinates[['x', 'y']]

    # using the KMeans method
    # tried Mean-shift and Spectral clustering, but the visualisation was poor
    # Mean Shift made confusing cluster areas and Spectral clustering had
    # sometimes overlapping or almost connecting clusters at some areas when
    # bigger number of clusters set (>8 cca)
    kmeans = sklearn.cluster.KMeans(n_clusters=12, n_init=12, random_state=0)\
        .fit(coords)
    working_gdf['cluster'] = kmeans.labels_

    cluster_counts = working_gdf['cluster'].value_counts()

    # set colors for number of points in clusters
    colormap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=cluster_counts.max())

    for cluster_num in range(kmeans.n_clusters):
        cluster = working_gdf[working_gdf['cluster'] == cluster_num]
        # get color for points of this cluster
        color = colormap(norm(cluster_counts[cluster_num]))
        # visualisation of cluster area
        if len(cluster) > 1:
            convex_hull = cluster.unary_union.convex_hull
            geopandas.GeoSeries([convex_hull], crs=working_gdf.crs)\
                .plot(ax=ax, alpha=0.5, color='grey')
        cluster.plot(ax=ax, markersize=2, color=color)

    # visual settings
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                        shrink=0.89, pad=0.02)
    cbar.set_label('Počet nehod v úseku')
    cbar.set_ticks(range(0, int(norm.vmax) + 1, 100))

    fig.tight_layout()
    if fig_location:
        fig.savefig(fig_location)

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)
