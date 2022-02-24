#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting

This submodule contains plotting functions for simple data visualization.

@author: Aaron Limoges
"""

import matplotlib.pyplot as plt
import seaborn as sns


def event_histogram(elements_dict: dict, element_mapping: dict, histplot_args: dict = None) -> plt.figure:
    """
    :param elements_dict:   an elements_dict, a dictionary where keys represent each unique element and
                            where each value a list of streak (i.e. interval) lengths
    :param element_mapping: A dictionary that maps each key of the elements_dict to a supplied string value.
                            Used only for generating figure titles.
    :param histplot_args:   A dictionary of accepted params for histplot.
    :return:                Returns a seaborn histplot of the counts of streak durations for each element
    """
    if histplot_args is None:
        histplot_args = {}

    for element, counts in elements_dict.items():
        sns.histplot(data=counts, **histplot_args).set_title(f'{element_mapping[element]}')
        plt.show()


def simple_hist(arr: list):
    fig = sns.histplot(data=arr)
    # print(arr)
    return fig


def xy_hist(df, cols):

    pass


if __name__ == '__main__':
    from src.chronio import event_intervals, streaks_to_lists
    import numpy as np
    import pandas as pd

    sns.set_theme()
    np.random.seed(5)
    arr = np.random.randint(0, 2, size=(100, 10))
    colnames = [f'Feature {i}' for i in range(0, arr.shape[1])]
    df = pd.DataFrame(arr, columns=colnames)

    print(df.head())
    print(df.columns[9])
    results = event_intervals(source_df=df, cols=[df.columns[9]])
    print(results['Feature 9'])
    to_plot = streaks_to_lists((results['Feature 9']))
    print(to_plot)

    element_map = {0: 'Miss',
                   1: 'Hit'}
    event_histogram(to_plot, element_mapping=element_map, histplot_args={'color': 'red'})

