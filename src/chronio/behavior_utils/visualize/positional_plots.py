#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Positional Plots

This submodule contains functions for plotting data related to location and position within an area.
For example, you may use this module to construct trackplots or heatmaps of position over time.

@author: Aaron Limoges
"""

import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def plot_positions(data: pd.DataFrame,
                   x_col: str,
                   y_col: str,
                   smoothen: bool = False,
                   downsample_factor: int = 1,
                   plot_args: dict = None):

    if plot_args is None:
        plot_args = {}

    x_coords = data[x_col].values
    y_coords = data[y_col].values

    x_coords = x_coords[0::downsample_factor]
    y_coords = y_coords[0::downsample_factor]

    if smoothen:
        from scipy.interpolate import splprep, splev

        def interpolate_polyline(polyline, num_points):
            duplicates = []
            for i in range(1, len(polyline)):
                if np.allclose(polyline[i], polyline[i - 1]):
                    duplicates.append(i)
            if duplicates:
                polyline = np.delete(polyline, duplicates, axis=0)
            tck, u = splprep(polyline.T, s=0)
            u = np.linspace(0.0, 1.0, num_points)
            return np.column_stack(splev(u, tck))
        x_coords = x_coords.values[~np.isnan(x_coords.values)]
        y_coords = y_coords.values[~np.isnan(y_coords.values)]

        dat = np.concatenate([x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)], axis=1)

        results = interpolate_polyline(dat, len(dat))
        x_coords = results[:, 0]
        y_coords = results[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.plot(x_coords, y_coords)
    return fig


def spatial_heatmap(spatial_data, q: float = 0.95, blur_sigma: float = 1, plot_args: dict = None):

    vmax = np.quantile(spatial_data[np.nonzero(spatial_data)], q=q)

    spatial_data = gaussian_filter(spatial_data, sigma=blur_sigma)

    if plot_args is None:
        fig = plt.imshow(spatial_data, vmax=vmax, cmap='inferno', interpolation='bicubic')

    else:
        fig = plt.imshow(spatial_data, vmax=vmax, **plot_args)
    return fig


def trial_trajectories(data: dict,
                       x_col: str,
                       y_col: str,
                       cmap: matplotlib.colors.Colormap = None,
                       plot_separate: bool = False):

    """
    :param data:            A dict whose keys represent names of trials (or sessions) and whose
                            keys are 2-D DataFrames where the columns represent the x and y
                            coordinates, respectively, throughout each trial.

    :param x_col:           String specifying the column containing x coordinates in the DataFrames

    :param y_col:           String specifying the column containing y coordinates in the DataFrames

    :param cmap:            If specified, the colormap to be used.
                            Right now, only lists of RGB tuples are supported.

    :param plot_separate:   If true, will plot all trials on separate subplots

    :return:
    """

    if cmap is None:
        cmap = sns.cubehelix_palette(n_colors=len(data.keys()))

    if plot_separate:
        # will produce a grid of trials with 3 columns by default
        # In future versions, consider allowing custom subplot dims
        n_cols = 3

        if len(data.keys()) % n_cols == 0:
            n_rows = len(data.keys()) // n_cols

        else:
            n_rows = len(data.keys()) // n_cols + 1

        print(n_rows)
        fig, axs = plt.subplots(n_rows, 3, figsize=(10, 10),
                                sharex=True, sharey=True)

        print(type(axs))

        for i, (key, df) in enumerate(data.items()):
            axs.flat[i].plot(df[x_col], df[y_col], color=cmap[i], linewidth=3)
            axs.flat[i].title.set_text(key)

        return fig, axs

    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        for i, df in enumerate(data.values()):
            ax.plot(df[x_col], df[y_col], color=cmap[i], linewidth=3)

        return fig, ax


if __name__ == '__main__':
    import pandas as pd
    from src.chronio import spatial_bins

    data = pd.read_csv('C://Users\\limogesaw\\Desktop\\mock_data\\Test_4.csv')

    fig = plot_positions(data, x_col='Centre position X', y_col='Centre position Y', smoothen=False,
                         downsample_factor=1)
    plt.show()

    H, x_edges, y_edges = spatial_bins(data, x_col='Centre position X', y_col='Centre position Y', bin_size=2,
                                       hist_kwargs={'normed': False})

    fig = spatial_heatmap(H, plot_args={'cmap': 'viridis', 'interpolation': 'bicubic'})
    plt.show()

    data_dict = {}
    for i in range(1, 10):
        data_dict[f'Trial {i}'] = data.iloc[100*i:100*i+100]

    print(data_dict.values())

    # Plot separately
    fig, axs = trial_trajectories(data_dict, x_col='Centre position X', y_col='Centre position Y',
                                  plot_separate=True)
    fig.show()

    # Plot together
    fig, ax = trial_trajectories(data_dict, x_col='Centre position X', y_col='Centre position Y')
    fig.show()
