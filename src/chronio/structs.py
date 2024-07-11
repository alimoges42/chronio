#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data structures for holding data and metadata associated with a specimen.
"""

from abc import ABC as _ABC
from typing import Any as _Any, List as _List
from collections.abc import Mapping as _Mapping
from copy import deepcopy
from pprint import pformat
import operator

import numpy as _np
import pandas as _pd
from scipy.interpolate import interp1d as _interp1d

import chronio.analyses as _analyses
from chronio.convention import Convention as _Convention
from chronio.io.exporters import _DataFrameExporter, _ArrayExporter
from chronio.logging_config import  setup_logging as _setup_logging
import chronio.exceptions as _exceptions

from chronio.utils import handle_inplace as _handle_inplace

__all__ = ['Metadata',
           'WindowPane',
           'Window',
           'EventData',
           'BehavioralTimeSeries',
           'NeuroTimeSeries',
           'DataCollection']

logger = _setup_logging()

class Metadata:
    """
    :param fpath:       Path to relevant file. For example, if a Metadata object is attached to a BehavioralTimeSeries,
                        fpath will refer to the location of the behavioral CSV, while if it is attached to a
                        SessionReference object, it will refer to the location of the SessionReference CSV file. Upon
                        instantiation, this parameter can be accessed via self.system['fpath'].
    :type fpath:        str

    :param stage_name:  Name of stage (if applicable). Upon instantiation, this parameter can be accessed via
                        self.session['stage'].
    :type stage_name:        str

    :param fps:         Frames per second (if applicable). Upon instantiation, this parameter can be accessed via
                        self.session['fps'], but note that it will be automatically updated if a downsampling method
                        is called.
    :type fps:          float

    :param trial_type:  Name of the trial type (if applicable). Upon instantiation, this parameter can be accessed via
                        self.window_params['trial_type'].
    :type trial_type:   str

    :param indices:     Indices that have been used as references for the window data (if applicable). Upon
                        instantiation, this parameter can be accessed via self.window_data['indices'].
    :type indices:      List[int]

    :param n_windows:   Number of windows (if applicable). Upon instantiation, this parameter can be accessed via
                        self.window_data['n_windows'].
    :type n_windows:    int

    :param meta_dict:   Additional metadata params. If supplied, will overwrite existing params for those attributes.
    :type meta_dict:    dict
    """

    def __init__(self,
                 fpath: str = None,

                 stage_name: str = None,
                 fps: float = None,

                 trial_type: str = None,
                 indices: _List[int] = None,
                 n_windows: int = None,

                 measure: str = None,

                 meta_dict: dict = {}
                 ):

        self.system = {'fpath': fpath}
        self.computed = {'fps': fps}
        self.session = {'stage_name': stage_name}
        self.window_params = {'indices': indices,
                              'n_windows': n_windows,
                              'trial_type': trial_type}
        self.pane_params = {'measure': measure}

        self.update(meta_dict=meta_dict)

    def __repr__(self):
        return f'{self.__class__}\n{pformat(self.__dict__)}'

    def __str__(self):
        return f'Metadata containing:\n{pformat(self.__dict__)}'

    def set_val(self,
                group: str,
                value_dict: dict):
        """
        Sets value of a specific metadata attribute (all metadata attributes are dicts).

        :param group:       The attribute to be updated. Accepted values are `'system'`, `'computed'`,
                            `'session'`, `'window_params'`, and `'pane_params'`.
        :type group:        str

        :param value_dict:  The new values of the attribute.
        :type value_dict:   dict
        """

        d = getattr(self, group)
        for key, value in value_dict.items():
            d[key] = value

    def update(self,
               meta_dict: dict):
        """
        Update the metadata by passing a dict where each key corresponds to an attribute, and values are the
        new dicts for that attribute.

        :param meta_dict:   dict containing metadata values
        :type: meta_dict:   dict
        """

        for group in list(self.__dict__.keys()):
            self.set_val(group, meta_dict)

    def export(self,
               convention: _Convention,
               function_kwargs: dict = None,
               **exporter_kwargs):
        # TODO: Think about how metadata should be exported
        pass


class _Structure(_ABC):

    def __init__(self,
                 data: _Any,
                 metadata: Metadata):
        self.data = data
        self.metadata = deepcopy(metadata)

    def export(self,
               convention: _Convention,
               **exporter_kwargs):
        """
        Parameters supplied by exporter_kwargs will replace those supplied by the convention object. This is to
        allow users on-the-fly editing without having to specify all the new fields of a convention object if they
        want to make a minor change to the convention.
        """

        # Overwrites convention params with user-specified export_kwargs (if supplied)
        export_kwargs = convention.get_params()
        export_kwargs.update(exporter_kwargs)
        export_kwargs['obj_type'] = self.__class__.__name__

        if isinstance(self.data, _np.ndarray):
            exporter = _ArrayExporter(obj=self.data, **export_kwargs)

        elif isinstance(self.data, _pd.DataFrame):
            # TODO: Support both CSV and XLSX io options - could be achieved by editing _DataFrameExporter?
            exporter = _DataFrameExporter(obj=self.data,
                                          metadata=self.metadata,
                                          **export_kwargs)

        else:
            raise ValueError(f'No export protocol for {type(self.data)} exists.')


class WindowPane(_Structure):
    """
    Class which represents the reduction of a :class: `chronio.structs.Window` on a single feature.
    The data contained in this class correspond to trial-by-trial data of a given feature.

    :param data:        a DataFrame of trial-by-trial data; output by `Window.collapse_on()`
    :type data:         pd.DataFrame

    :param metadata:    metadata for the object
    :type metadata:     :class: `chronio.structs.Metadata`

    :param pane_params: information about the settings used to obtain the pane.
                        These are then passed to `self.metadata.pane_params`
    :type pane_params:  dict
    """
    def __init__(self,
                 data: _pd.DataFrame,
                 metadata: Metadata,
                 pane_params: dict = None):
        super().__init__(data=data, metadata=metadata)

        if not pane_params:
            pane_params = {}

        for key, value in pane_params.items():
            self.metadata.pane_params[key] = pane_params[key]

    def __str__(self):
        return f'{self.__class__.__name__} object of shape {self.data.shape}'

    def __repr__(self):
        return f'{self.__class__.__name__}(num_rows={self.data.shape[0]}, num_columns={self.data.shape[1]})'


    def event_counts(self, axis: int = 0) -> _np.ndarray:
        """
        Return counts of events for binarized data.

        :param axis: axis along which to compute. By default this is set to 0, which will count the events in each
                     trial across the window.
        :return: Returns the total number of events of interest along a given axis in the window
        """
        return _analyses.count_events(self.data.values, axis=axis)


class Window(_Structure):
    """
        Class which represents the trial-by-trial data for a given trial of all features
        contained in a time series csv.

        :param data:            a list of DataFrames of trial-by-trial data
        :type data:             List[pd.DataFrame]

        :param metadata:        metadata for the object
        :type metadata:         :class: `chronio.structs.Metadata`

        :param window_params:   information about the settings used to obtain the pane.
                                These are then passed to `self.metadata.window_params`
        :type window_params:    dict
        """
    def __init__(self,
                 data: _List[_pd.DataFrame],
                 metadata: Metadata,
                 window_params: dict = None):
        super().__init__(data=data, metadata=metadata)

        if not window_params:
            window_params = {}

        if all(window.shape == self.data[0].shape for window in self.data):
            window_params['dims'] = self.data[0].shape
        else:
            raise ValueError("Each DataFrame must have equal shape.")

        window_params.update({'n_windows': len(self.data)})

        print(f'{window_params = }')
        self.metadata.set_val('window_params', window_params)


    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"object containing {self.metadata.window_params['n_windows']} " \
               f"windows of shape {self.metadata.window_params['dims']}"

    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"num_windows={self.metadata.window_params['n_windows']}, " \
               f"dims={self.metadata.window_params['dims']})"

    def collapse_on(self,
                    feature: str) -> WindowPane:
        """
        :param feature:     variable (i.e. column name) from which you would like to extract data

        :return:            WindowPane of the specified feature
        """
        _columns = [f'Trial_{i}' for i in range(1, len(self.data) + 1)]

        agg = _pd.DataFrame(_np.vstack([window[feature].values for window in self.data]).T,
                            columns=_columns,
                            index=self.data[0].index)

        stacked = WindowPane(data=agg,
                             metadata=self.metadata,
                             pane_params={'measure': feature})
        return stacked


class _TimeSeries(_Structure):

    def __init__(self,
                 data: _Any = None,
                 metadata: Metadata = Metadata(),
                 fpath: str = None,
                 time_col: str = None,
                 read_csv_kwargs: dict = None):

        logger.info("Initializing _TimeSeries object")

        try:
            super().__init__(data=data, metadata=metadata)

            if not read_csv_kwargs:
                read_csv_kwargs = {}

            if fpath:
                logger.info(f"Loading data from file: {fpath}")
                self.metadata.system['fpath'] = fpath
                self.data = _pd.read_csv(self.metadata.system['fpath'], **read_csv_kwargs)

            if time_col:
                logger.info(f"Setting time column: {time_col}")
                self.data.set_index(time_col, inplace=True)

            self._time_col = time_col
            self._update_fps()
            logger.info("_TimeSeries object initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing _TimeSeries: {str(e)}")
            raise _exceptions.ProcessingError("Failed to initialize _TimeSeries object") from e

    def _update_fps(self):
        try:
            time_range = self.data.index.values[-1] - self.data.index.values[0]
            self.metadata.computed['fps'] = round(self.data.shape[0] / time_range, 2)
            logger.debug(f"Updated fps: {self.metadata.computed['fps']}")
        except Exception as e:
            logger.error(f"Error updating fps: {str(e)}")
            raise _exceptions.ProcessingError("Failed to update fps") from e

    def set_time_col(self, time_col):
        self._time_col = time_col
        self.data.set_index(time_col, inplace=True)
        logger.info("Time column set to: {time_col}")

    @_handle_inplace
    def downsample_to_length(self, length: int, method: str = 'nearest'):
        """
        Downsample the dataset to a target length.

        :param length: Desired length of downsampled results
        :param method: Method of downsampling. Valid methods are 'nearest', 'linear', and 'cubic'.
        :return: A new _TimeSeries object with the downsampled data if inplace=False,
                 otherwise None and the current object is modified.
        """
        return _analyses.downsample_to_length(self.data, length, method)

    @_handle_inplace
    def downsample_by_time(self, interval: float, method: str = 'mean', round_time: int = None):
        """
        Downsample a dataset by a specified time interval.

        :param interval:    Binning interval (in seconds).
        :param method:      Aggregation method. The 'mean' method will compute the mean value for each interval.
        :param round_time:  If provided, round the new timestamps to this number of decimal places
        :return:            A new _TimeSeries object with the downsampled data if inplace=False,
                            otherwise None and the current object is modified.
        """
        logger.info(f"Downsampling data by time. Interval: {interval}, Method: {method}")
        try:
            if not isinstance(interval, (int, float)):
                raise _exceptions.InputError("Interval must be a number")
            if method not in ['mean', 'min', 'max']:
                raise _exceptions.InputError(f"Invalid method '{method}'. Valid methods are 'mean', 'min', and 'max'")

            binned = _analyses.downsample_by_time(self.data, interval, method, round_time)

            logger.info("Downsampling by time completed successfully")
            return binned

        except _exceptions.InputError as e:
            logger.error(f"Input error in downsample_by_time: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error in downsample_by_time: {str(e)}")
            raise _exceptions.ProcessingError("An error occurred during downsampling") from e

    @_handle_inplace
    def threshold(self,
                  thr: float,
                  direction: str = '>',
                  binarize: bool = False,
                  columns: list = None):
        """
        Set a threshold on the specified columns of the dataset and, if desired, binarize the resulting dataset.

        :param thr:       Threshold value
        :param direction: Comparison operator ('>', '>=', '==', '!=', '<', '<=')
        :param binarize:  If True, binarize the result
        :param columns:   List of columns to be thresholded. If None, all columns are used.
        :return:          Thresholded _TimeSeries object if inplace=False, otherwise None
        """
        return _analyses.threshold_data(self.data, thr, direction, binarize, columns)

    @_handle_inplace
    def binarize(self,
                 columns: list = None,
                 method: str = 'round'):
        """
        Binarize a dataset. All nonzero entries (including negative numbers) will be set to 1.

        :param columns: Columns to binarize. If None, all columns are used.
        :param method:  Binarization method. Valid methods are 'round', 'retain_ones', 'retain_zeros'
        :return:        Binarized _TimeSeries object if inplace=False, otherwise None
        """
        return _analyses.binarize_data(self.data, columns, method)

    @_handle_inplace
    def normalize(self,
                  baseline: list = None,
                  columns: list = None):
        """
        Normalize a dataset to some window. Normalization is achieved via a z-scoring-like method,
        wherein a window may be established as a baseline mean. If no window is specified, the entire time
        series is used as the baseline mean.

        :param baseline: A list of two indices, where `baseline[0]` designates beginning of window,
                         and `baseline[1]` designates the end of the window.
        :param columns:  A list which, if supplied, applies normalization only to the specified columns.
        :return:         Normalized _TimeSeries object if inplace=False, otherwise None
        """
        return _analyses.normalize_data(self.data, baseline, columns)

    def split_by_trial(self,
                       timepoints: list,
                       trial_type: str = None,
                       pre_period: float = 0,
                       post_period: float = 0,
                       center_index: bool = True) -> Window:

        """
        :param timepoints:      Timepoints to align to

        :param trial_type:      User-defined name of trial type

        :param pre_period:      Time (in seconds) desired to obtain prior to trial onset

        :param post_period:     Time (in seconds) desired to obtain following end of trial

        :param center_index:    If True, reset index of each window to be centered around the
                                timepoint that window was aligned to.

        :return:                List of aligned trial data
        """
        indices = [round(i * self.metadata.computed['fps']) for i in timepoints]

        trials = Window(data=_analyses.windows_aligned(source_df=self.data,
                                                       fps=self.metadata.computed['fps'],
                                                       alignment_points=indices,
                                                       pre_frames=int(pre_period * self.metadata.computed['fps']),
                                                       post_frames=int(post_period * self.metadata.computed['fps']),
                                                       center_index=center_index),
                        metadata=self.metadata,
                        window_params={'trial_type': trial_type})

        return trials

    def get_events(self,
                   columns: _List[str] = None,
                   get_intervals: bool = True) -> dict:

        if columns is None:
            columns = self.data.columns

        events = _analyses.get_events(source_df=self.data, cols=columns, get_intervals=get_intervals)

        events_dict = {}
        for col, event in events.items():
            events_dict[col] = EventData(data=event, metadata=self.metadata)

        return events_dict


class BehavioralTimeSeries(_TimeSeries):

    """
    Time series that represents behavioral data.

    The BehavioralTimeSeries class, unlike the NeuroTimeSeries class, does not assume that all columns
    represent the same form of data, nor that each column is expressed in the same units.
    """
    def __init__(self,
                 data: _Any = None,
                 metadata: Metadata = Metadata(),
                 fpath: str = None,
                 time_col: str = None):
        super().__init__(data=data, metadata=metadata, fpath=fpath, time_col=time_col)

    def compute_velocity(self, x_col: str, y_col: str):
        pass


class NeuroTimeSeries(_TimeSeries):

    """
    Time series that represents some form of neural activity data.

    Data in the NeuroTimeSeries class are assumed to be homogeneous. For example, each column may represent a neuron's
    activity, a channel from a recording array, etc.
    """

    def __init__(self,
                 data: _Any = None,
                 metadata: Metadata = Metadata(),
                 fpath: str = None,
                 time_col: str = None):
        super().__init__(data=data, metadata=metadata, fpath=fpath, time_col=time_col)

    def correlate(self, axis='cells', method='spearman') -> _Structure:
        methods = ['spearman', 'pearson']
        axes = ['cells', 'time']

        if method not in methods:
            print(f'Correlation method {method} not currently supported.'
                  f'Supported methods are {methods}.')

        if method == 'spearman':
            # TODO: spearman correlation
            pass

        elif method == 'pearson':
            # TODO: pearson correlation
            pass

        return _Structure(data=None, metadata=self.metadata)


class EventData(_Structure):
    def __init__(self,
                 data: _pd.DataFrame = None,
                 metadata: Metadata = Metadata(),
                 fpath: str = None):
        # Check to make sure input DataFrame contains correct columns
        # if data.columns != ['epoch type', 'epoch onset', 'epoch end', 'epoch duration']:
        #   raise ValueError('Input parameter "data" must be a DataFrame with columns matching: '
        #                    '["epoch type", "epoch onset", "epoch end", "epoch duration"].')
        super().__init__(data=data, metadata=metadata)

        if fpath:
            self.metadata.system['fpath'] = fpath

            self.data = _pd.read_csv(self.metadata.system['fpath'])

    def to_times(self,
                 fps: float = None,
                 inplace: bool = False):
        if fps is None:
            fps = self.metadata.computed['fps']
        df = self.data
        df['epoch onset'] = _analyses.frames_to_times(fps, df['epoch onset'].values)
        df['epoch end'] = _analyses.frames_to_times(fps, df['epoch end'].values)
        df['epoch duration'] = _analyses.frames_to_times(fps, df['epoch duration'].values)

        if inplace:
            self.data = df

        else:
            return self.__class__(data=df, metadata=self.metadata)

    def events_during(self,
                      windows: _List[list],
                      hanging_events: str = 'none',
                      inplace=False):
        """
        Select events only during specific windows.

        :param windows:
        :param hanging_events:
        :param inplace:
        :return:
        """
        valid_hanging_events = ['start', 'end', 'both', 'none']

        if hanging_events not in valid_hanging_events:
            raise ValueError(f'Invalid input for hanging_events. Valid inputs are {valid_hanging_events}.')

        dfs = []

        for tup in windows:

            if hanging_events in 'start':
                df = self.data[self.data['epoch end'].between(tup[0], tup[1])]

            elif hanging_events in 'end':
                df = self.data[self.data['epoch onset'].between(tup[0], tup[1])]

            elif hanging_events in 'both':
                df = self.data[self.data['epoch end'].between(tup[0], tup[1])]
                df2 = self.data[self.data['epoch onset'].between(tup[0], tup[1])]

                if len(df.index) > 0:
                    idx_start = df.index[0]
                else:
                    idx_start = 0

                df = df.merge(df2, how='outer')
                df.index = [i + idx_start for i in range(0, len(df))]

            elif hanging_events in 'none':
                df = self.data[self.data['epoch onset'].between(tup[0], tup[1])]
                df2 = self.data[self.data['epoch end'].between(tup[0], tup[1])]

                if len(df.index) > 0:
                    idx_start = df.index[0]
                else:
                    idx_start = 0

                df = df.merge(df2, how='inner')
                df.index = [i + idx_start for i in range(0, len(df))]

            else:
                df = _pd.DataFrame()

            if df.shape[0] > 0:
                dfs.append(df)

        if len(dfs) > 0:
            df = _pd.concat(dfs).drop_duplicates()

        else:
            df = _pd.DataFrame()

        if inplace:
            self.data = df

        else:
            return self.__class__(data=df, metadata=self.metadata)


class DataCollection(_Mapping):
    def __init__(self,
                 datasets: _Any,
                 metadata: Metadata = None,
                 field_mapper: str = None):
        """

        :param datasets:        If list, ...
                                If dict, ...
        :type datasets:         List or dict

        :param metadata:        Metadata
        :type metadata:         :class: `chronio.structs.Metadata`

        :param field_mapper:    string that specifies the metadata field to be mapped
        :type field_mapper:     str
        """
        datasets_ = {}
        if isinstance(datasets, list):
            for d in datasets:
                if isinstance(d, _Structure):
                    datasets_[d.metadata[field_mapper]] = d
                else:
                    raise TypeError(f'Type {type(d)} is not supported for DataCollection object.')

        elif isinstance(datasets, dict):
            for name, d in datasets.items():
                if isinstance(d, _Structure):
                    datasets_[name] = d
                else:
                    raise TypeError(f'Type {type(d)} is not supported for DataCollection object.')

        self.datasets = datasets_
        self.metadata = metadata

    def __getitem__(self, key):
        return self.datasets[key]

    def __iter__(self):
        return iter(self.datasets)

    def __len__(self):
        return len(self.datasets)

    def __repr__(self):
        return f"{type(self).__name__}({self.datasets})"

    def export(self,
               convention: _Convention,
               subset: list = None,
               **exporter_kwargs):
        """
        Parameters supplied by exporter_kwargs will replace those supplied by the convention object. This is to
        allow users on-the-fly editing without having to specify all the new fields of a convention object if they
        want to make a minor change to the convention.
        """

        # Overwrites convention params with user-specified export_kwargs (if supplied)
        export_kwargs = convention.get_params()
        export_kwargs.update(exporter_kwargs)
        export_kwargs['obj_type'] = self.__class__.__name__

        for name, dataset in self.datasets.items():
            if name in subset:
                if type(dataset.data) == _np.ndarray:
                    exporter = _ArrayExporter(obj=dataset.data, **export_kwargs)

                elif type(dataset.data) == _pd.DataFrame:
                    # TODO: Support both CSV and XLSX io options - could be achieved by editing _DataFrameExporter?
                    exporter = _DataFrameExporter(obj=dataset.data,
                                                  metadata=dataset.metadata,
                                                  **export_kwargs)

                else:
                    raise ValueError(f'No export protocol for {type(dataset.data)} exists.')


