#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic tools for retrieving information on event lengths, intervals, and onset times,
as well as window alignment.

Author: Aaron Limoges
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import groupby
from typing import List


__all__ = ['frames_to_times',
           'times_to_frames',
           'event_onsets',
           'event_intervals',
           'streaks_to_lists',
           'spatial_bins',
           'get_state_durations',
           'windows_aligned',
           'windows_custom']


def frames_to_times(fps: float, frame_numbers: list) -> list:
    """
    Retrieve timestamps given frame numbers.

    :param fps: frame rate (frames per second)
    :param frame_numbers:
    :return: list of timestamps that each frame corresponds to.
    """

    return list(map(lambda x: x / fps, frame_numbers))


def times_to_frames(fps: float, timestamps: list) -> list:
    """
    Retrieve frame numbers given timestamps or time intervals.

    :param fps: frame rate (frames per second)
    :param timestamps:
    :return: list of frames that each timestamp corresponds to.
    """

    return list(map(lambda x: int(x * fps), timestamps))


def event_onsets(source_df: pd.DataFrame,
                 cols: list) -> dict:
    """
    Obtain the frame numbers that signify onsets of events in desired columns of events.

    :param source_df:   original dataframe to use

    :param cols:        list of columns of the dataframe for which you would like IEIs to be computed

    :return:            dict of dicts containing the frame numbers corresponding to each onset of an event
                        for each specified column
    """
    onsets = {}
    for col in cols:

        onsets[col] = {}

        col_states = source_df[col].to_frame()
        col_states['streak_start'] = col_states[col].ne(col_states[col].shift()).astype(str)
        col_states = col_states.groupby(col)

        for state, group in col_states:
            onsets[col][state] = group[group['streak_start'] == 'True'].index.values

    return onsets


def event_intervals(source_df: pd.DataFrame,
                    cols: list) -> dict:
    """
    Compute the inter-event intervals (IEIs) for desired columns of a DataFrame.

    :param source_df:   original dataframe to use

    :param cols:        list of columns of the dataframe for which you would like IEIs to be computed


    :return:            dict of event interval data for each specified column
    """

    results = defaultdict(pd.DataFrame)

    for col in cols:
        data = {'Elements': [], 'Counts': []}

        streaks = [list(group) for key, group in groupby(source_df[col].values.tolist())]

        for streak in streaks:
            data['Elements'].append(streak[0])
            data['Counts'].append(len(streak))

        results[col] = pd.DataFrame.from_dict(data=data)

    return results


def streaks_to_lists(streak_df: pd.DataFrame) -> dict:
    """
    Convert streaks to lists.

    :param streak_df:   A DataFrame of streaks with the column names of 'Elements' and 'Streaks'.
                        These correspond to the DataFrames contained in the dict values returned by
                        the event_intervals function.


    :return:            Returns elements_dict, a dictionary of each unique element found within the
                        'Elements' column of the streak_df. Each value is simply the value originally
                        contained within the 'Counts' column.
    """

    elements_dict = {}
    for uniq in streak_df['Elements'].unique():
        elements_dict[uniq] = streak_df.loc[streak_df['Elements'] == uniq]['Counts'].tolist()

    return elements_dict


def spatial_bins(source_df: pd.DataFrame,
                 x_col: str,
                 y_col: str,
                 bin_size: float = 1,
                 handle_nans: str = 'bfill',
                 hist_kwargs: dict = None
                 ) -> tuple:

    """
    Retrieve the spatial bins of events

    :param source_df:   df containing columns for x and y coordinates

    :param x_col:       name of the column with x data

    :param y_col:       name of the column with y data

    :param bin_size:    bin size (in mm)

    :param handle_nans: valid options are 'bfill', 'ffill', or 'drop'

    :param hist_kwargs: optional kwargs for np.histogram2d


    :return:            tuple of H, x_edges, and y_edges (from np.histogram2d)
    """

    if hist_kwargs is None:
        hist_kwargs = {}

    if handle_nans == 'drop':
        source_df = source_df.dropna()

    else:
        source_df = source_df.fillna(method=handle_nans)

    # Compute number of x and y bins needed at the current bin_size
    _x_bins = (source_df[x_col].max() - source_df[x_col].min()) / bin_size
    x_bins = int((_x_bins + bin_size) - (_x_bins % bin_size))

    _y_bins = (source_df[y_col].max() - source_df[y_col].min()) / bin_size
    y_bins = int((_y_bins + bin_size) - (_y_bins % bin_size))

    # Extract 2D histogram
    H, x_edges, y_edges = np.histogram2d(source_df[x_col].values,
                                         source_df[y_col].values,
                                         bins=[x_bins, y_bins],
                                         **hist_kwargs)
    H = H.T

    return H, x_edges, y_edges


def get_state_durations(source_df: pd.DataFrame,
                        cols: List[str],
                        values: list = None,
                        fps: float = 30) -> pd.DataFrame:
    """
    Computes the durations of states in specific columns of a DataFrame.

    :param source_df:   DataFrame

    :param cols:        Columns for which the durations are to be computed

    :param values:      Values for which duration will be computed. If None, will extract duration for all states

    :param fps:         Imaging rate (frames per second) of df

    :return:            DataFrame whose columns are equivalent to "cols" parameter and whose rows are either
                            1) a list of corresponding to "values" parameter (if provided), or
                            2) all unique states found across all columns.

                        Each element is the duration of each state (in seconds).

    """

    durations = pd.DataFrame()

    for col in cols:
        col_durations = source_df[col].value_counts() / fps

        if values:
            durations[col] = col_durations[values]

        else:
            durations[col] = col_durations

    return durations


def windows_custom(source_df: pd.DataFrame, startpoints: list, endpoints: list) -> list:
    """
    Grab the windows between each pair of startpoints and endpoints.

    Note that this function allows window sizes to vary in length.

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
    Grab windows aligned in reference to a set of alignment points.

    User may specify the number of frames to be included before and after each alignment point so that each window
    in the returned list is of equal length.

    This function is useful for analyzing data relative to the onset of specific events or trials.

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