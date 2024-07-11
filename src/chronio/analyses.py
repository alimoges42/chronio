#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic tools for retrieving information on event lengths, intervals, and onset times,
as well as window alignment.

Author: Aaron Limoges
"""

from typing import List, Dict, Any
from typing import Callable as _Callable
import operator as _operator

import pandas as _pd
import numpy as _np
from scipy.ndimage import label as _label
from scipy.interpolate import interp1d as _interp1d


__all__ = ['segment_df',
           'frames_to_times',
           'times_to_frames',
           'count_events',
           'get_events',
           'spatial_bins',
           'windows_aligned',
           'windows_custom',
           'reconstruct_time',
           'downsample_to_length',
           'downsample_by_time',]


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


def spatial_bins(source_df: _pd.DataFrame,
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

    :param hist_kwargs: optional kwargs for _np.histogram2d


    :return:            tuple of H, x_edges, and y_edges (from _np.histogram2d)
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
    H, x_edges, y_edges = _np.histogram2d(source_df[x_col].values,
                                         source_df[y_col].values,
                                         bins=[x_bins, y_bins],
                                         **hist_kwargs)
    H = H.T

    return H, x_edges, y_edges


def windows_custom(source_df: _pd.DataFrame, startpoints: list, endpoints: list) -> list:
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


def windows_aligned(source_df:          _pd.DataFrame,
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

        if end > len(source_df):
            raise ValueError(f'End frame number {end} exceeds {len(source_df) = }.')
        windows.append(source_df.loc[source_df.index.values[start:end:1]])

    if center_index:
        centered_index = _np.round(_np.linspace(-(pre_frames - 1)/fps, post_frames/fps, num=len(windows[0])), 2)

        for window in windows:
            window.set_index([centered_index], inplace=True)

    return windows


def reconstruct_time(trial_onsets: Dict[str, list],
                     trial_durations: Dict[str, Any],
                     stage_duration: float,
                     fps: float = 1):
    """
    Cast known event times to a pandas DataFrame.

    :param trial_onsets:    A dict whose keys are names of trial types and whose values are a list of onset times
                            (in seconds) for that trial type
    :type trial_onsets:     Dict[str, list]

    :param trial_durations: A dict whose keys are names of trial types and whose values are the durations (in seconds)
                            for each trial type
    :type trial_durations:  Dict[str, Any]

    :param stage_duration:  Total duration of the stage (in seconds)
    :type stage_duration:   float

    :param fps:             Frame rate (frames per second)
    :type fps:              float

    :return:                A t x n _pd.DataFrame, where t represents timestamps x fps, and n represents to the number
                            of trial types.
    """

    if sorted(trial_onsets.keys()) != sorted(trial_durations.keys()):
        raise ValueError('Keys of inputs dicts do not match.')

    num_features = len(trial_onsets.keys())

    _arr = _np.zeros((stage_duration * fps, num_features))

    for i, (trial_type, onsets) in enumerate(trial_onsets.items()):
        trial_dur = trial_durations[trial_type]
        print(onsets)
        for onset in onsets:
            _arr[onset * fps: (onset + trial_dur) * fps, i] = 1

    _arr = _pd.DataFrame(_arr, columns=trial_onsets.keys(), index=_np.arange(0, len(_arr)))

    return _arr


def count_events(data: _np.ndarray, axis: int = 0) -> _np.ndarray:
    """
    Count the number of events in binarized data.

    :param data: A numpy array of binarized data where 1s represent event occurrences.
    :param axis: The axis along which to compute. Default is 0.
    :return: An array containing the total number of events along the specified axis.
    """
    diff = _np.diff(data, axis=axis)
    diff[diff == -1] = 0
    return diff.sum(axis=axis)


def get_events(source_df: _pd.DataFrame,
               cols: List[str] = None,
               get_intervals: bool = True) -> dict:
    """
    Find event onsets, ends, and durations for a given dataframe. This will only work on columns
    of the DataFrame that contain binary values. It is assumed that 1s correspond to the occurrence of
    an event, while 0s indicate the lack of an event. This function uses a connected components algorithm to
    identify events and from there will record the onset, end, and duration of each event within a column.

    :param source_df:       Original DataFrame to get events from.
    :type source_df:        _pd.DataFrame

    :param cols:            If provided, events will be obtained only for those columns.
                            If not provided, the function will get events from all columns of source_df.
    :type cols:             List[str]

    :param get_intervals:   If True, also return information on each interval between events in the
                            resulting DataFrame.
    :type get_intervals:    bool

    :return:                dict whose keys correspond to each column in cols parameter and whose
                            values are DataFrames of events (and intervals if get_intervals=True).
    """

    if cols is None:
        cols = source_df.columns

    results = {}

    for col in cols:
        event_comps, num_events = _label(source_df[col].values)

        events_dict = {'epoch type': [],
                       'epoch onset': [],
                       'epoch end': [],
                       'epoch duration': []}

        for i in range(1, num_events + 1):
            events = _np.argwhere(event_comps == i)
            event_onset = events[0]
            event_end = events[-1]
            event_dur = event_end - event_onset + 1

            events_dict['epoch type'].append('event')
            events_dict['epoch onset'].append(*event_onset)
            events_dict['epoch end'].append(*event_end)
            events_dict['epoch duration'].append(*event_dur)

        if get_intervals:
            interval_comps, num_intervals = _label(_np.abs(source_df[col].values - 1))
            intervals_dict = {'epoch type': [],
                              'epoch onset': [],
                              'epoch end': [],
                              'epoch duration': []}

            for i in range(1, num_intervals + 1):
                intervals = _np.argwhere(interval_comps == i)
                interval_onset = intervals[0]
                interval_end = intervals[-1]
                interval_dur = interval_end - interval_onset + 1

                intervals_dict['epoch type'].append('interval')
                intervals_dict['epoch onset'].append(*interval_onset)
                intervals_dict['epoch end'].append(*interval_end)
                intervals_dict['epoch duration'].append(*interval_dur)

            events_df = _pd.DataFrame(events_dict)
            intervals_df = _pd.DataFrame(intervals_dict)
            merged_df = _pd.concat([events_df, intervals_df], axis=0)

            results[col] = merged_df

        else:
            events_df = _pd.DataFrame(events_dict)
            results[col] = events_df

    return results


def downsample_by_time(data: _pd.DataFrame,
                       interval: float,
                       method: str = 'mean',
                       round_time: int = None,
                       update_fps:_Callable = None):
    """
    Downsample a dataset by a specified time interval.

    :param interval:    Binning interval (in seconds).

    :param method:      Aggregation method. The 'mean' method will compute the mean value for each interval.

    :param inplace:     if True, updates the self.data parameter to contain only the downsampled dataset.
    """

    bins = _np.arange(data.index.values[0] - interval, data.index.values[-1] + interval, interval)

    # Currently supported methods
    methods = ['mean', 'min', 'max']

    if method == 'mean':
        binned = data.groupby(_pd.cut(data.index, bins)).mean()

    elif method == 'min':
        binned = data.groupby(_pd.cut(data.index, bins)).min()

    elif method == 'max':
        binned = data.groupby(_pd.cut(data.index, bins)).max()

    else:
        raise ValueError(f'Method {method} not recognized. Currently accepted methods are {methods}')

    binned.index = _pd.Series(binned.index).apply(lambda x: x.left).astype(float)
    binned.index = binned.index + interval

    if round_time:
        binned.index = _np.round(binned.index, round_time)

    if update_fps:
        update_fps()

    return binned


def downsample_to_length(data: _pd.DataFrame,
                         length: int,
                         method: str = 'nearest',
                         update_fps: _Callable = None) -> _pd.DataFrame:
    """
    Downsample a dataset to a target length.

    :param data: Input DataFrame with time index
    :param length: Desired length of downsampled results
    :param method: Method of downsampling. Valid methods are 'nearest', 'linear', and 'cubic'.
    :return: Downsampled DataFrame
    """
    if method not in ['nearest', 'linear', 'cubic']:
        raise ValueError(f"Invalid method '{method}'. Valid methods are 'nearest', 'linear', and 'cubic'.")

    x = _np.arange(len(data))
    f = _interp1d(x, data.values, axis=0, kind=method)
    x_new = _np.linspace(0, len(data) - 1, length)
    new_values = f(x_new)

    new_index = _pd.Index(_np.linspace(data.index[0], data.index[-1], length))

    if update_fps:
        update_fps()
    ds = _pd.DataFrame(new_values, index=new_index, columns=data.columns)

    return ds


def threshold_data(data: _pd.DataFrame,
                   thr: float,
                   direction: str = '>',
                   binarize: bool = False,
                   columns: list = None) -> _pd.DataFrame:
    """
    Set a threshold on the specified columns of the dataset and, if desired, binarize the resulting dataset.

    :param data:      DataFrame to threshold
    :param thr:       Threshold value
    :param direction: Comparison operator ('>', '>=', '==', '!=', '<', '<=')
    :param binarize:  If True, binarize the result
    :param columns:   List of columns to be thresholded. If None, all columns are used.
    :return:          Thresholded DataFrame
    """
    ops = {'>': _operator.gt, '>=': _operator.ge, '==': _operator.eq,
           '!=': _operator.ne, '<': _operator.lt, '<=': _operator.le}

    if direction not in ops.keys():
        raise ValueError(f'Invalid operation {direction}. Valid operations are {list(ops.keys())}')

    if columns is None:
        columns = data.columns

    result = data.copy()
    result[columns] = result[columns].where(ops[direction](result[columns], thr), other=0)

    if binarize:
        result[columns] = result[columns].astype(bool).astype(int)

    return result


def binarize_data(data: _pd.DataFrame,
                  columns: list = None,
                  method: str = 'round') -> _pd.DataFrame:
    """
    Binarize a dataset. All nonzero entries (including negative numbers) will be set to 1.

    :param data:    DataFrame to binarize
    :param columns: Columns to binarize. If None, all columns are used.
    :param method:  Binarization method. Valid methods are 'round', 'retain_ones', 'retain_zeros'
    :return:        Binarized DataFrame
    """
    if columns is None:
        columns = data.columns

    result = data.copy()

    if method == 'round':
        result.loc[:, columns] = result.loc[:, columns].fillna(method='bfill')
        result.loc[:, columns] = result.loc[:, columns].clip(0, 1)
        result.loc[:, columns] = result.loc[:, columns].round(0).astype(_np.int16)
    elif method == 'retain_ones':
        result.loc[:, columns] = result.loc[:, columns].clip(0, 1)
        result.loc[:, columns] = result.loc[:, columns].where(result.loc[:, columns] == 1, 0).astype(_np.int16)
    elif method == 'retain_zeros':
        result.loc[:, columns] = result.loc[:, columns].clip(0, 1)
        result.loc[:, columns] = result.loc[:, columns].where(result.loc[:, columns] == 0, 1).astype(_np.int16)
    else:
        valid_methods = ['round', 'retain_ones', 'retain_zeros']
        raise ValueError(f'Specified method not supported. Valid methods are {valid_methods}.')

    return result


def normalize_data(data: _pd.DataFrame,
                   baseline: list = None,
                   columns: list = None) -> _pd.DataFrame:
    """
    Normalize a dataset to some window. Normalization is achieved via a z-scoring-like method,
    wherein a window may be established as a baseline mean. If no window is specified, the entire time
    series is used as the baseline mean.

    :param data:     DataFrame to normalize
    :param baseline: A list of two indices, where `baseline[0]` designates beginning of window,
                     and `baseline[1]` designates the end of the window.
    :param columns:  A list which, if supplied, applies normalization only to the specified columns.
    :return:         Normalized DataFrame
    """
    if columns is None:
        columns = data.columns

    result = data.copy()
    to_normalize = result[columns]

    if baseline:
        mean = _np.nanmean(to_normalize.values[baseline[0]: baseline[1]], axis=0)
        std = _np.nanstd(to_normalize.values[baseline[0]: baseline[1]], axis=0)
    else:
        mean = _np.nanmean(to_normalize.values, axis=0)
        std = _np.nanstd(to_normalize.values, axis=0)

    result[columns] = (to_normalize - mean) / std

    return result


def segment_df(source_df: _pd.DataFrame, startpoints: list, endpoints: list):
    """
    Given a list of startpoints and endpoints, segment a DataFrame into a list of DataFrames.

    :param source_df:   DataFrame to segment
    :type source_df:    _pd.DataFrame

    :param startpoints: List of indices that determine onset of epochs
    :type startpoints:  list

    :param endpoints:   List of indices that determine ending of epochs
    :type endpoints:    list

    :return:            tuple of lists, with the first list corresponding to windows between start and endpoints,
                        and the second list containing windows outside of that (i.e. the masked windows).
    """
    mask = _binary_array(startpoints=startpoints, endpoints=endpoints, duration=source_df.shape[0], width=1, frate=1)

    source_df['Mask'] = mask
    windows = [g[g.columns[:-1]][g.Mask != 0] for k, g in source_df.groupby((source_df.Mask == 0).cumsum()) if len(g) > 1]
    inverted = [g[g.columns[:-1]][g.Mask != 1] for k, g in source_df.groupby((source_df.Mask == 1).cumsum()) if len(g) > 1]
    source_df.drop(columns=['Mask'], inplace=True)

    return windows, inverted


def _binary_array(startpoints: _np.ndarray, endpoints: _np.ndarray, duration: float, width: int, frate: float = 1):
    """
    Create an array of 1s and 0s, where 1s correspond to events and 0s correspond to lack of events.

    :param startpoints: Points at which each event starts
    :type startpoints:  list

    :param endpoints:   Points at which each event ends
    :type endpoints:    list

    :param duration:    Total duration of the desired array (in seconds)
    :type duration:     float

    :param width:       Number of columns the array should contain. Each column will contain the same values.
    :type width:        int

    :param frate:       frame rate (frames per second)
    :type frate:        float

    :return:            _np.array where number of rows is determined by frate * duration, and number of columns by width.
    """

    fps = int(1 / frate)
    startpoints = startpoints * fps
    endpoints = endpoints * fps

    on_start = 0 if startpoints[0] != 0 else 1
    on_end = 0 if endpoints[-1] != duration * frate else 1

    events = []
    intervals = []

    if not on_start:
        intervals.append(_np.zeros(((startpoints[0]), width), int))

    for start, end in zip(startpoints, endpoints):
        events.append(_np.ones((end - start, width), int))

    for start, end in zip(startpoints[1:], endpoints[:-1]):
        intervals.append(_np.zeros((start - end, width), int))

    if not on_end:
        intervals.append(_np.zeros((duration * fps - endpoints[-1], width), int))

    result = [None] * (len(events) + len(intervals))

    if on_start:
        result[::2] = events
        result[1::2] = intervals

    else:
        result[::2] = intervals
        result[1::2] = events

    result = _np.concatenate(result).reshape(width, duration * fps)[0]
    return result

if __name__ == '__main__':
    df = _pd.DataFrame(data=_np.random.randint(0, 2, (100, 3)), columns=['Col1', 'Col2', 'Col3'])
    results = get_events(df, cols=['Col1', 'Col3'], get_intervals=True)
    print(results)
