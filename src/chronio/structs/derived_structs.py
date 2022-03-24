#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This contains two special classes for working with window data, i.e. data of a consistent length
across an arbitrary number of repeated trials.


Author: Aaron Limoges
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
    """
    Base class for structures that are derived from raw structures
    """
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
    """
    Class which represents the reduction of a :class: `chronio.structs.derived_structs.Window` on a single feature.
    The data contained in this class correspond to trial-by-trial data of a given feature.

    :param data:        a DataFrame of trial-by-trial data; output by `Window.collapse_on()`
    :type data:         pd.DataFrame

    :param metadata:    metadata for the object
    :type metadata:     :class: `chronio.structs.metadata.Metadata`

    :param pane_params: information about the settings used to obtain the pane.
                        These are then passed to `self.metadata.pane_params`
    :type pane_params:  dict
    """
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
    """
        Class which represents the trial-by-trial data for a given trial of all features
        contained in a time series csv.

        :param data:            a list of DataFrames of trial-by-trial data
        :type data:             List[pd.DataFrame]

        :param metadata:        metadata for the object
        :type metadata:         :class: `chronio.structs.metadata.Metadata`

        :param window_params:   information about the settings used to obtain the pane.
                                These are then passed to `self.metadata.window_params`
        :type window_params:    dict
        """
    def __init__(self,
                 data: _List[_pd.DataFrame],
                 metadata: _Metadata,
                 window_params: dict = None):

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