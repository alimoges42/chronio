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


def simple_hist(arr: list):
    fig = sns.histplot(data=arr)
    # print(arr)
    return fig


def xy_hist(df, cols):

    pass
