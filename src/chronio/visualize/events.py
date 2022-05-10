#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting

This submodule contains plotting functions for simple data visualization.

@author: Aaron Limoges
"""
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def event_histogram(elements_dict: dict, element_mapping: dict, histplot_args: dict = None) -> plt.figure:
    """
    :param elements_dict:   an elements_dict, a dictionary where keys represent each unique element and
                            where each value a list of streak (i.e. interval) lengths
    :type elements_dict:    dict

    :param element_mapping: A dictionary that maps each key of the elements_dict to a supplied string value.
                            Used only for generating figure titles.
    :type element_mapping:  dict

    :param histplot_args:   A dictionary of accepted params for histplot.
    :type histplot_args:    dict

    :return:                Returns a seaborn histplot of the counts of streak durations for each element
    """
    if histplot_args is None:
        histplot_args = {}

    for element, counts in elements_dict.items():
        sns.histplot(data=counts, **histplot_args).set_title(f'{element_mapping[element]}')
        plt.show()


def plot_events(coords_dict: Dict[str, List[tuple]],
                sort_order: list = None,
                fps: float = 1,
                figsize: tuple = None,
                cmap: Any = 'Set1',
                frame_nums: bool = True):
    """
    :param coords_dict: Dict whose keys represent trial or event names and whose values are lists containing tuples of
                        length 2, where each tuple specifies the start times and end times (in seconds) of each event.
    :type coords_dict:  Dict[str, List[tuple]]

    :param sort_order:  If provided, a list specifying the order in which the events will be sorted on the plot. Events
                        the beginning of the list will appear at the bottom of the plot.
    :type sort_order:   list

    :param fps:         Frame rate (frames per second)
    :type fps:          float

    :param figsize:     Size of the figure. This param is passed to plt.subplots().
    :type figsize:      tuple

    :param cmap:        If cmap is supplied as a string, this function will load that cmap from matplotlib.
                        Alternatively, a cmap or a list of iterable colors can be supplied directly.
    :type cmap:         Any

    :param frame_nums:  if True, plot the frame numbers as a dual x-axis at the top of the figure
    :type frame_nums:   bool

    :return:
    """
    if not figsize:
        figsize = (15, len(coords_dict.keys()) / 5)

    fig, ax1 = plt.subplots(figsize=figsize)

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    if not sort_order:
        sort_order = coords_dict.keys()

    for i, ttype in enumerate(sort_order):
        for xs in coords_dict[ttype]:
            ys = [ttype, ttype]

            if hasattr(cmap, 'colors'):
                ax1.plot(xs, ys, color=cmap.colors[i], label=ttype, zorder=1)
                ax1.scatter(x=xs[0], y=ttype, color=cmap.colors[i], edgecolors=cmap.colors[i], zorder=2)
                ax1.scatter(x=xs[1], y=ttype, color='white', edgecolors=cmap.colors[i], zorder=3)

            else:
                ax1.plot(xs, ys, color=cmap[i], label=ttype, zorder=1)
                ax1.scatter(x=xs[0], y=ttype, color=cmap[i], edgecolors=cmap[i], zorder=2)
                ax1.scatter(x=xs[1], y=ttype, color='white', edgecolors=cmap[i], zorder=3)

    ax1.spines["left"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)

    ax1.set_xlabel(f"Time (s)", size=14)

    ax2 = ax1.twiny()
    ax2.spines["left"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    def tick_function(x_coords):
        x_new = [int(x * fps) for x in x_coords]
        return [z for z in x_new]

    tick_locations = ax1.get_xticks()

    ax2.set_xticks(tick_locations)
    ax2.set_xticklabels(tick_function(tick_locations))
    ax2.set_xlabel(f"Frame number (at {fps :.2f} fps)", size=14)

    # Note that the call to ax1.get_xlim() must come after the ax2.set_xticks() call (as seen just a few lines above).
    ax2.set_xlim(ax1.get_xlim())

    axs = [ax1]
    if frame_nums:
        axs.append(ax2)

    return fig, axs


def simple_hist(arr: list):
    fig = sns.histplot(data=arr)
    # print(arr)
    return fig

