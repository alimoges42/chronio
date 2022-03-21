#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
derived_structs.py

This contains two special classes for working with window data, i.e. data of a consistent length
across an arbitrary number of repeated trials.


@author: Aaron Limoges
"""

from abc import ABC as _ABC
from typing import Any as _Any, List as _List
import numpy as _np
import pandas as _pd
from scipy.stats import sem as _sem

from chronio.design.convention import Convention as _Convention
from chronio.export.exporter import _DataFrameExporter, _ArrayExporter
from chronio.structs.metadata import Metadata as _Metadata


class _DerivedStructure(_ABC):

    def __init__(self, data: _Any, metadata: _Metadata):
        self.data = data
        self.metadata = metadata

    def export(self, convention: _Convention, function_kwargs: dict = None, **exporter_kwargs):
        """
        Parameters supplied by exporter_kwargs will replace those supplied by the convention object. This is to
        allow users on-the-fly editing without having to specify all the new fields of a convention object if they
        want to make a minor change to the convention.
        """

        # Overwrites convention params with user-specified export_kwargs (if supplied)
        export_kwargs = convention.get_params()
        export_kwargs.update(exporter_kwargs)

        if type(self.data) == _np.ndarray:
            exporter = _ArrayExporter(obj=self.data, **export_kwargs)

        elif type(self.data) == _pd.DataFrame:
            # TODO: Support both CSV and XLSX export options - could be achieved by editing _DataFrameExporter?
            exporter = _DataFrameExporter(obj=self.data, **export_kwargs)

        else:
            raise ValueError(f'No export protocol for {type(self.data)} exists.')

        if function_kwargs:
            exporter.export(self.data, **function_kwargs)

        else:
            exporter.export(self.data)


class WindowPane(_DerivedStructure):

    def __init__(self, data: _pd.DataFrame, metadata: _Metadata, pane_params: dict = None):
        super().__init__(data=data, metadata=metadata)

        if not pane_params:
            pane_params = {}

        for key, value in pane_params:
            setattr(self.metadata.pane_params, key, pane_params[key])

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


class Window(_DerivedStructure):

    def __init__(self,
                 data: _List[_pd.DataFrame],
                 metadata: _Metadata,
                 window_params: dict = None):
        """
        :param data:
        :param metadata:
        :param window_params:
        """
        super().__init__(data=data, metadata=metadata)

        if not window_params:
            window_params = {}

        for key, value in window_params:
            setattr(self.metadata.window_params, key, window_params[key])

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
                             metadata=self.metadata)
        return stacked

    def save_windows(self, save_params: dict):

        pass


if __name__ == '__main__':
    from chronio.structs.raw_structs import BehavioralTimeSeries

    my_series = BehavioralTimeSeries(fpath='C://Users\\limogesaw\\Desktop\\mock_data\\Test_4.csv')
    my_trials = my_series.split_by_trial(trial_type='so', pre_period=5, post_period=30, indices=[100, 400, 700, 1000])

    print(my_trials.data)
    print(str(my_trials))
    print(repr(my_trials))
    print(my_trials.data[0].columns)
    collapsed = my_trials.collapse_on('Speed')

    #collapsed.data = np.random.randint(2, size=(100, 4))
    print(collapsed.data)
    print(str(collapsed))
    print(repr(collapsed))

    #print(collapsed.event_counts())

    print(collapsed.data.index.shape)

    t = _pd.to_timedelta(collapsed.data.index, unit='seconds')
    s = collapsed.data.set_index(t).resample('1S').last().reset_index(drop=True)
    print(t)
    print(s)
    print(collapsed.metadata.session['fps'])