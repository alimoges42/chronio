#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Slicing Tools

This submodule contains tools for slicing and indexing timestamped data.
For example, it supports grabbing defined windows of consistent lengths around given target points.

@author: Aaron Limoges
"""

import numpy as np
import pandas as pd
from .conversions import frames_to_times


def windows_custom(source_df: pd.DataFrame, startpoints: list, endpoints: list) -> list:
    """
    Grabs each window between each pair of startpoints and endpoints, permitting variable window sizes.

    :param source_df: original array to grab frames from
    :param startpoints: list of frame numbers where each window should start
    :param endpoints: list of frame numbers where each window should end
    :return: list of dataframes with each df representing a single window
    """

    # Check that startpoints and endpoints match in length
    if len(startpoints) != len(endpoints):
        raise ValueError('Number of startpoints and endpoints must be equal.')

    windows = []
    for start, end in zip(startpoints, endpoints):
        if end < start:
            raise ValueError(f'Cannot slice. Endpoint {end} is smaller than startpoint {start}.')
        windows.append(source_df[start:end:1])

    return windows


def windows_aligned(source_df:          pd.DataFrame,
                    fps:                float,
                    alignment_points:   list,
                    pre_frames:         int = 0,
                    post_frames:        int = 0,
                    center_index:       bool = True) -> list:
    """
    Grabs a window aligned in reference to a set of alignment points. User may specify
    the number of frames to be included before and after each alignment point so that each window
    in the returned list is of equal length.

    This function is useful for analyzing data relative to the onset of specific events or cues.

    Note that if the number of pre_period or post_period frames leads to a certain window that extends outside
    the available data, that specific window will be returned as an empty dataframe.


    :param source_df:           Original array to grab frames from

    :param fps:                 Frames per second

    :param alignment_points:    List of frame numbers where each window should start

    :param pre_frames:          Number of frames before each alignment point to be included

    :param post_frames:         Number of frames after each alignment point to be included

    :param center_index:        If True, set the index at each alignment point to zero.
                                Indices at timepoints/frames before each alignment point
                                will then be set to negative values.

                                If False, the indices around each alignment point will be
                                retained separately.

    :return:                    List of dataframes with each df representing a single window
    """

    startpoints, endpoints = list(), list()
    for point in alignment_points:
        startpoints.append(point - pre_frames)
        endpoints.append(point + post_frames)

    windows = []
    for start, end in zip(startpoints, endpoints):
        if end < start:
            raise ValueError(f'Cannot slice. Endpoint {end} is smaller than startpoint {start}.')
        windows.append(source_df[start:end:1])

    if center_index:
        centered_index = frames_to_times(fps=fps, frame_numbers=windows[0].index - alignment_points[0])

        for window in windows:
            window.set_index([centered_index], inplace=True)

    return windows


if __name__ == '__main__':
    np.random.seed(5)
    arr = np.random.randint(0, 2, size=(100, 10))
    df = pd.DataFrame(arr)

    print(df.head())

    windows = windows_custom(df, startpoints=[1, 10, 20], endpoints=[4, 15, 40])
    for window in windows:
        print(window)

    windows = windows_aligned(df, alignment_points=[1, 10, 21], pre_frames=5, post_frames=5)
    for window in windows:
        print(window)
    print(windows[0])
    print(len(windows))
