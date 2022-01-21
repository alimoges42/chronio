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
from typing import List
from matplotlib.colors import ListedColormap
from dataclasses import dataclass


@dataclass
class StackedWindow:

    fps: float
    data: pd.DataFrame

    def mean(self):
        return np.mean(self.data, axis=1)

    def sterr(self):
        return sem(self.data, axis=1)

    def stdev(self):
        return np.std(self.data, axis=1)

    def plot(self):
        pass


class WindowData:
    def __init__(self, windows: List[pd.DataFrame], fps: float, indices: list,
                 trial_type: str = None):
        """
        :param trial_type:
        :param windows: list of DataFrames aligned with reference to some value.
        """
        self.windows = windows
        self.num_windows = len(windows)
        self.fps = fps
        self.indices = indices
        self.trial_type = trial_type

        if all(x.shape == windows[0].shape for x in windows):
            self.dims = windows[0].shape

        else:
            raise ValueError("Each DataFrame must have equal shape.")

    def __str__(self):
        return f'WindowData object containing {self.num_windows} windows of shape {self.dims}'

    def __repr__(self):
        return f'WindowData(num_windows={self.num_windows}, dims={self.dims})'

    def collapse_on(self, column: str) -> StackedWindow:
        """
        :param column: variable from which you would like to extract data
        :return:
        """
        _columns = [f'Trial_{i}' for i in range(1, len(self.windows) + 1)]

        agg = pd.DataFrame(np.vstack([window[column].values for window in self.windows]),
                           columns=self.windows[0].index,
                           index=_columns)

        stacked = StackedWindow(data=agg,
                                fps=self.fps)
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
    from time_series import IndividualTimeSeries

    my_series = IndividualTimeSeries('C://Users\\limogesaw\\Desktop\\mock_data\\Test_4.csv', fps=8)
    my_trials = my_series.split_by_trial(trial_type='so', pre_period=5, post_period=10)

    print(my_trials.windows)
    print(str(my_trials))
    print(repr(my_trials))