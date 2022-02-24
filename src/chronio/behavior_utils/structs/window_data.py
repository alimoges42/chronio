#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Window Data

This contains two special classes for working with window data, i.e. data of a consistent length
across an arbitrary number of repeated trials.


@author: Aaron Limoges
"""

import pandas as pd
import numpy as np
from scipy.stats import sem
from typing import List, Any
from matplotlib.colors import ListedColormap
from .structure import _DataStructure


class StackedWindow(_DataStructure):

    def __init__(self, fps: float, data: pd.DataFrame, metadata: Any):
        super().__init__(data, metadata)
        self.fps = fps
        self.data = data

    def __str__(self):
        return f'{self.__class__.__name__} object of shape {self.data.shape}'

    def __repr__(self):
        return f'{self.__class__.__name__}(num_rows={self.data.shape[0]}, num_columns={self.data.shape[1]})'

    def mean(self, axis: int = 1):
        """
        :param axis:    axis along which to compute. By default this is set to 1, which will return the mean
                        value at each timepoint across the window.

        :return:        Returns the mean of the data along the specified axis (passed to np.mean())
        """

        return np.mean(self.data, axis=axis)

    def sem(self, axis: int = 1):
        """
        :param axis:    axis along which to compute. By default this is set to 1, which will return the standard error
                        at each timepoint across the window.

        :return:        Returns the standard error of the mean along the specified axis (passed to scipy.stats.sem())
        """

        return sem(self.data, axis=axis)

    def std(self, axis: int = 1):
        """
        :param axis:    axis along which to compute. By default this is set to 1, which will return the standard
                        deviation at each timepoint across the window.

        :return:        Returns the standard deviation along the specified axis (passed to np.std())
        """
        return np.std(self.data, axis=axis)

    def plot(self):
        pass


class WindowData(_DataStructure):
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

    def collapse_on(self, feature: str) -> StackedWindow:
        """
        :param feature:     variable (i.e. column name) from which you would like to extract data

        :return:            StackedWindow of the specified feature
        """
        _columns = [f'Trial_{i}' for i in range(1, len(self.data) + 1)]

        agg = pd.DataFrame(np.vstack([window[feature].values for window in self.data]),
                           columns=self.data[0].index,
                           index=_columns)

        stacked = StackedWindow(data=agg,
                                fps=self.fps,
                                metadata=None)
        return stacked

    def spatial_heatmap(self, trials: list = None):

        if trials is None:
            # Plot all trials
            pass
        else:
            for trial in trials:
                # Plot trial(s)
                pass
        pass

    def trajectories_by_trial(self, trial: int = None, cmap: ListedColormap = 'viridis'):
        pass

    def survival_curve(self):
        pass

    def save_windows(self, save_params: dict):

        pass


if __name__ == '__main__':
    from time_series import BehavioralTimeSeries

    my_series = BehavioralTimeSeries('C://Users\\limogesaw\\Desktop\\mock_data\\Test_4.csv', fps=8)
    my_trials = my_series.split_by_trial(trial_type='so', pre_period=5, post_period=10)

    print(my_trials.windows)
    print(str(my_trials))
    print(repr(my_trials))
