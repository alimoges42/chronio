#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
windows.py

This contains two special classes for working with window data, i.e. data of a consistent length
across an arbitrary number of repeated trials.


@author: Aaron Limoges
"""

from typing import List, Any
import pandas as pd
import numpy as np
from scipy.stats import sem

from chronio.structs.structure import _DataStructure


class WindowPane(_DataStructure):

    def __init__(self, fps: float, data: pd.DataFrame, metadata: Any):
        super().__init__(data, metadata)
        self.fps = fps
        self.data = data

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

        return sem(self.data, axis=axis)

    def event_counts(self, axis: int = 0):
        """
        Return counts of events for binarized data. Note that this method requires the variable of interest to
        consist solely of 1s and 0s, where 1s represent event occurrence.

        :param axis:    axis along which to compute. By default this is set to 0, which will count the events in each
                        trial across the window.

        :return:        Returns the total number of events of interest along a given axis in the window
        """

        diff_ = np.diff(self.data, axis=axis)
        diff_[diff_ == -1] = 0
        return diff_.sum(axis=axis)

    def plot(self):
        pass


class Window(_DataStructure):
    def __init__(self, data: List[pd.DataFrame], fps: float, indices: list, metadata: Any,
                 trial_type: str = None):
        """
        :param windows:     list of DataFrames aligned with reference to some value.

        :param trial_type:  user-specified name for trial type being extracted
        """

        super().__init__(data, metadata)
        self.data = data
        self.num_windows = len(data)
        self.fps = fps
        self.indices = indices
        self.trial_type = trial_type

        if all(window.shape == data[0].shape for window in data):
            self.dims = data[0].shape

        else:
            raise ValueError("Each DataFrame must have equal shape.")

    def __str__(self):
        return f'{self.__class__.__name__} object containing {self.num_windows} windows of shape {self.dims}'

    def __repr__(self):
        return f'{self.__class__.__name__}(num_windows={self.num_windows}, dims={self.dims})'

    def collapse_on(self, feature: str) -> WindowPane:
        """
        :param feature:     variable (i.e. column name) from which you would like to extract data

        :return:            WindowPane of the specified feature
        """
        _columns = [f'Trial_{i}' for i in range(1, len(self.data) + 1)]

        agg = pd.DataFrame(np.vstack([window[feature].values for window in self.data]).T,
                           columns=_columns,
                           index=self.data[0].index)

        stacked = WindowPane(data=agg,
                             fps=self.fps,
                             metadata=None)
        return stacked

    def save_windows(self, save_params: dict):

        pass


if __name__ == '__main__':
    from behavior import BehavioralTimeSeries

    my_series = BehavioralTimeSeries('C://Users\\limogesaw\\Desktop\\mock_data\\Test_4.csv')
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

    t = pd.to_timedelta(collapsed.data.index, unit='seconds')
    s = collapsed.data.set_index(t).resample('1S').last().reset_index(drop=True)
    print(t)
    print(s)
