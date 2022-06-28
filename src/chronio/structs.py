#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data structures for holding data and metadata associated with a specimen.
"""

from abc import ABC as _ABC
from typing import Any as _Any, List as _List
from collections.abc import Mapping as _Mapping, Iterable as _Iterable

import numpy as _np
import pandas as _pd
import pandas as pd
from scipy.interpolate import interp1d as _interp1d
from scipy.stats import sem as _sem

import chronio.analyses as _analyses
from chronio.convention import Convention as _Convention
from chronio.io.exporters import _DataFrameExporter, _ArrayExporter

__all__ = ['Metadata',
           'WindowPane',
           'Window',
           'EventData',
           'BehavioralTimeSeries',
           'NeuroTimeSeries']


class Metadata:
    """
    :param fpath:       Path to relevant file. For example, if a Metadata object is attached to a BehavioralTimeSeries,
                        fpath will refer to the location of the behavioral CSV, while if it is attached to a
                        SessionReference object, it will refer to the location of the SessionReference CSV file. Upon
                        instantiation, this parameter can be accessed via self.system['fpath'].
    :type fpath:        str

    :param stage_name:       Name of stage (if applicable). Upon instantiation, this parameter can be accessed via
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

    def set_val(self, group: str, value_dict: dict):
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

    def update(self, meta_dict: dict):
        """
        Update the metadata by passing a dict where each key corresponds to an attribute, and values are the
        new dicts for that attribute.

        :param meta_dict:   dict containing metadata values
        :type: meta_dict:   dict
        """

        for group in list(self.__dict__.keys()):
            self.set_val(group, meta_dict)

    def export(self, convention: _Convention, function_kwargs: dict = None, **exporter_kwargs):
        # TODO: Think about how metadata should be exported
        pass


class _Structure(_ABC):

    def __init__(self, data: _Any, metadata: Metadata):
        self.data = data
        self.metadata = metadata

    def export(self, convention: _Convention, **exporter_kwargs):
        """
        Parameters supplied by exporter_kwargs will replace those supplied by the convention object. This is to
        allow users on-the-fly editing without having to specify all the new fields of a convention object if they
        want to make a minor change to the convention.
        """

        # Overwrites convention params with user-specified export_kwargs (if supplied)
        export_kwargs = convention.get_params()
        export_kwargs.update(exporter_kwargs)
        export_kwargs['obj_type'] = self.__class__.__name__

        if type(self.data) == _np.ndarray:
            exporter = _ArrayExporter(obj=self.data, **export_kwargs)

        elif type(self.data) == _pd.DataFrame:
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
    def __init__(self, data: _pd.DataFrame, metadata: Metadata, pane_params: dict = None):
        super().__init__(data=data, metadata=metadata)

        if not pane_params:
            pane_params = {}

        for key, value in pane_params.items():
            self.metadata.pane_params[key] = pane_params[key]

    def __str__(self):
        return f'{self.__class__.__name__} object of shape {self.data.shape}'

    def __repr__(self):
        return f'{self.__class__.__name__}(num_rows={self.data.shape[0]}, num_columns={self.data.shape[1]})'

    def sem(self, axis: int = 0):
        """
        :param axis:    axis along which to compute. By default this is set to 0, which will return the standard error
                        at each timepoint across the window.

        :return:        Returns the standard error of the mean along the specified axis (passed to scipy.stats.sem())
        """

        return _sem(self.data, axis=axis)

    def event_counts(self, axis: int = 0):
        """
        Return counts of events for binarized data. Note that this method requires the variable of interest to
        consist solely of 1s and 0s, where 1s represent event occurrence.

        :param axis:    axis along which to compute. By default this is set to 0, which will count the events in each
                        trial across the window.

        :return:        Returns the total number of events of interest along a given axis in the window
        """

        diff_ = _np.diff(self.data, axis=axis)
        diff_[diff_ == -1] = 0
        return diff_.sum(axis=axis)


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

        for key, value in window_params.items():
            self.metadata.window_params[key] = value

        self.metadata.window_params['n_windows'] = len(self.data)

        if all(window.shape == self.data[0].shape for window in self.data):
            self.metadata.window_params['dims'] = self.data[0].shape

        else:
            raise ValueError("Each DataFrame must have equal shape.")

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"object containing {self.metadata.window_params['n_windows']} " \
               f"windows of shape {self.metadata.window_params['dims']}"

    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"num_windows={self.metadata.window_params['n_windows']}, " \
               f"dims={self.metadata.window_params['dims']})"

    def collapse_on(self, feature: str) -> WindowPane:
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

    def __init__(self, data: _Any, metadata: Metadata, fpath: str = None, time_col: str = 'Time',
                 read_csv_kwargs: dict = None):
        super().__init__(data=data, metadata=metadata)

        if not read_csv_kwargs:
            read_csv_kwargs = {}

        if fpath:
            self.metadata.system['fpath'] = fpath

            self.data = _pd.read_csv(self.metadata.system['fpath'], **read_csv_kwargs)

        self._time_col = time_col
        self._update_fps()

    def _update_fps(self):
        self.metadata.computed['fps'] = round(self.data.shape[0] / self.data[self._time_col].values[-1], 2)

    def downsample_to_length(self, method: str = 'nearest', length: int = None, inplace=False):
        """
        Downsample a dataset to a target length

        :param method:      method of downsampling. Valid methods are 'nearest', 'linear', and 'cut'.
                            For 'nearest' and 'linear' methods, np.interp1d is used, and these methods are passed to
                            the 'kind' parameter of np.interp1d.

        :param length:      desired length of downsampled results

        :param inplace:     if True, updates the self.data parameter to contain only the downsampled dataset.
        """

        x = self.data.index
        y = self.data.values

        f = _interp1d(x, y, axis=0, kind=method)
        x_new = _np.linspace(x[0], x[-1], num=length)
        y_new = f(x_new)
        idx = _np.arange(0, len(y_new), 1)

        df = _pd.DataFrame(index=idx, columns=self.data.columns, data=y_new)

        if inplace:
            self.data = df
            self._update_fps()

        else:
            return self.__class__(data=df, metadata=self.metadata, time_col=self._time_col)

    def downsample_by_time(self, interval: float, method: str = 'mean', inplace=False):
        """
        Downsample a dataset by a specified time interval.

        :param interval:    Binning interval (in seconds).

        :param method:      Aggregation method. The 'mean' method will compute the mean value for each interval.

        :param inplace:     if True, updates the self.data parameter to contain only the downsampled dataset.
        """

        # Currently supported methods
        methods = ['mean']

        bins = _np.arange(0, self.data[self._time_col].values[-1] + 1, interval)

        if method == 'mean':
            binned = self.data.groupby(_pd.cut(self.data[self._time_col], bins)).mean()

        else:
            raise ValueError(f'Method {method} not recognized. Currently accepted methods are {methods}')

        binned.drop(columns=[self._time_col], inplace=True)
        binned.reset_index(inplace=True)
        binned[self._time_col] = _np.arange(self.data[self._time_col].values[0], self.data[self._time_col].values[-1] - interval + 1, interval)
        binned = binned[binned[self._time_col] <= self.data[self._time_col].values[-1]]

        if inplace:
            self.data = binned
            self._update_fps()
            return None

        else:
            return self.__class__(data=binned, metadata=self.metadata, time_col=self._time_col)

    def get_events(self, cols: _List[str] = None, get_intervals: bool = True) -> dict:
        events = _analyses.get_events(source_df=self.data, cols=cols, get_intervals=get_intervals)

        events_dict = {}
        for col, event in events.items():
            events_dict[col] = EventData(data=event, metadata=self.metadata)

        return events_dict

    def threshold(self, thr: float, binarize: bool = False, columns: _List[str] = None, inplace: bool = False):
        """
        Set a threshold on the specified columns of the dataset and, if desired, binarize the resulting dataset.
        Currently, the same threshold is applied to all specified columns, so if separate thresholds are needed for
        different columns, then this method should be run repeatedly, specifying the target columns for each threshold.

        :param thr:         Threshold value.

        :param binarize:    If True, follow the thresholding with a binarization step that sets all nonzero values to 1.

        :param columns:     List of columns to be thresholded.

        :param inplace:     If True, updates the self.data parameter to contain only the thresholded (or, if applicable,
                            the binarized) dataset.


        :returns:           If inplace == False, returns thresholded (or, if applicable, the binarized) dataset.
                            If inplace == True, modify inplace and return None
        """

        if inplace:
            self.data[columns][self.data[columns] < thr] = 0

            if binarize:
                self.binarize(columns=columns, inplace=inplace)

            return None

        else:
            data = self.data.copy(deep=True)
            data[columns][data[columns] < thr] = 0

            # Can't use self.binarize() here because we made a deep copy
            if binarize:
                data[columns][data[columns] >= thr] = 1

            return self.__class__(data=data, metadata=self.metadata, time_col=self._time_col)

    def binarize(self, columns: _List[str] = None, method: str = 'round', inplace: bool = False):
        """
        Binarize a dataset. All nonzero entries (including negative numbers!!) will be set to 1.

        :param method:

        :param columns: Columns to binarize.

        :param inplace: If True, updates the self.data parameter to contain only the binarized dataset.

        :returns:           If inplace == False, returns the binarized dataset.
                            If inplace == True, modify inplace and return None
        """

        if columns is None:
            columns = self.data.columns

        data = self.data.copy(deep=True)

        if method == 'round':
            data.loc[:, columns] = data.loc[:, columns].round(0)
            data[columns] = data.loc[:, columns].astype(_np.int16)

        elif method == 'hold_ones':
            data.loc[:, columns][data.loc[:, columns] != 1] = 0
            data[columns] = data.loc[:, columns].astype(_np.int16)

        elif method == 'hold_zeros':
            data.loc[:, columns][data.loc[:, columns] != 0] = 1
            data[columns] = data.loc[:, columns].astype(_np.int16)

        else:
            valid_methods = ['round', 'hold_ones', 'hold_zeros']
            raise ValueError(f'Specified method not supported. Valid methods are {valid_methods}.')

        if inplace:
            self.data = data
            return None

        else:
            return self.__class__(data=data, metadata=self.metadata, time_col=self._time_col)

    def split_by_trial(self,
                       indices: list,
                       trial_type: str = None,
                       pre_period: float = 0,
                       post_period: float = 0,
                       storage_params: str = None) -> Window:

        """
        :param indices:         Indices to align to

        :param trial_type:      User-defined name of trial type

        :param pre_period:      Time (in seconds) desired to obtain prior to trial onset

        :param post_period:     Time (in seconds) desired to obtain following end of trial

        :param storage_params:  Dict of parameters for saving

        :return:                List of aligned trial data
        """
        trials = Window(data=_analyses.windows_aligned(source_df=self.data,
                                                       fps=self.metadata.computed['fps'],
                                                       alignment_points=indices,
                                                       pre_frames=int(pre_period * self.metadata.computed['fps']),
                                                       post_frames=int(post_period * self.metadata.computed['fps'])),
                        metadata=self.metadata,
                        window_params={'trial_type': trial_type})

        if storage_params:
            pass

        return trials

    def normalize(self, window: _List[int] = None, columns: _List[str] = None, inplace=False):
        """
        Normalize a dataset to some window. Normalization is achieved via a z-scoring-like method,
        wherein a window may be established as a baseline mean. If no window is specified, the entire time
        series is used as the baseline mean.

        :param window:  A list of two indices, where `window[0]` designates beginning of window,
                        and `window[1]` designates the end of the window.
        :type window:   List[int]

        :param columns: A list which, if supplied, applies normalization only to the specified columns.
        :type columns:  List[str]

        :param inplace: If True, update `self.data` with new values. If False, return a new object.
        :type inplace:  bool

        :return:        Derivative TimeSeries class.
        """

        if columns:
            data = self.data[columns]
        else:
            data = self.data

        if window:
            mean = _np.nanmean(data.values[window[0]: window[1]])
            std = _np.nanstd(data.values[window[0]: window[1]])

        else:
            mean = _np.nanmean(data.values)
            std = _np.nanstd(data.values)

        z = (data.values - mean) / std
        z = _pd.DataFrame(z, columns=data.columns, index=data.index)

        if inplace:
            self.data = z
            return None

        else:
            return self.__class__(data=z, metadata=self.metadata, time_col=self._time_col)


class BehavioralTimeSeries(_TimeSeries):

    """
    Time series that represents behavioral data.

    The BehavioralTimeSeries class, unlike the NeuroTimeSeries class, does not assume that all columns
    represent the same form of data, nor that each column is expressed in the same units.
    """
    def __init__(self, data: _Any = None, metadata: Metadata = Metadata(), fpath: str = None, time_col: str = 'Time'):
        super().__init__(data=data, metadata=metadata, fpath=fpath, time_col=time_col)

    def compute_velocity(self, x_col: str, y_col: str):
        pass


class NeuroTimeSeries(_TimeSeries):

    """
    Time series that represents some form of neural activity data.

    Data in the NeuroTimeSeries class are assumed to be homogeneous. For example, each column may represent a neuron's
    activity, a channel from a recording array, etc.
    """

    def __init__(self, data: _Any = None, metadata: Metadata = Metadata(), fpath: str = None, time_col: str = 'Time'):
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
    def __init__(self, data: pd.DataFrame = None, metadata: Metadata = Metadata(), fpath: str = None):
        # Check to make sure input DataFrame contains correct columns
        # if data.columns != ['epoch type', 'epoch onset', 'epoch end', 'epoch duration']:
        #   raise ValueError('Input parameter "data" must be a DataFrame with columns matching: '
        #                    '["epoch type", "epoch onset", "epoch end", "epoch duration"].')
        super().__init__(data=data, metadata=metadata)

        if fpath:
            self.metadata.system['fpath'] = fpath

            self.data = _pd.read_csv(self.metadata.system['fpath'])

    def to_times(self, fps: float = None, inplace: bool = False):
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

    def events_during(self, windows: _List[list], hanging_events: str = 'none', inplace=False):
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
            print(tup)

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
                df = pd.DataFrame()

            if df.shape[0] > 0:
                dfs.append(df)
        df = pd.concat(dfs).drop_duplicates()

        if inplace:
            self.data = df

        else:
            return self.__class__(data=df, metadata=self.metadata)


class DataCollection(_Mapping):
    def __init__(self, datasets: _Any, metadata: Metadata = None, field_mapper: str = None):
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
        if type(datasets) == list:
            for d in datasets:
                if type(d) == _Structure:
                    datasets_[d.metadata[field_mapper]] = d
                else:
                    raise TypeError(f'Type {type(d)} is not supported for DataCollection object.')

        elif type(datasets) == dict:
            for name, d in datasets.items:
                if type(d) == _Structure:
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

    def export(self, convention: _Convention, subset: list = None, **exporter_kwargs):
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


