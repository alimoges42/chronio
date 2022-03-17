#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
behavior.py

This submodule contains classes for working with behavioral time series data.
It is useful for storing the raw time series dataset as well as its metadata.

@author: Aaron Limoges
"""

from typing import List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from chronio.structs.windows import Window
from chronio.process.analyses import event_onsets, event_intervals, streaks_to_lists, windows_aligned


@dataclass
class BehavioralTimeSeries:
    fpath: str

    def __post_init__(self):
        self.data = pd.read_csv(self.fpath)
        self.fps = self.data.shape[0] / round(self.data['Time'].values[-1], 0)

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

        f = interp1d(x, y, axis=0, kind=method)
        x_new = np.linspace(x[0], x[-1], num=length)
        y_new = f(x_new)

        if inplace:
            self.data = pd.DataFrame(index=x_new, columns=self.data.columns, data=y_new)

        else:
            return pd.DataFrame(index=x_new, columns=self.data.columns, data=y_new)

    def downsample_by_time(self, interval: float, method: str = 'mean'):
        """
        Downsample a dataset by a specified time interval.

        :param interval:    Binning interval (in seconds).

        :param method:      Aggregation method. The 'mean' method will compute the mean value for each interval.
        """

        # Currently supported methods
        methods = ['mean']

        bins = np.arange(0, self.data['Time'].values[-1], interval)

        if method == 'mean':
            binned = self.data.groupby(pd.cut(self.data["Time"], bins)).mean()

        else:
            raise ValueError(f'Method {method} not recognized. Currently accepted methods are {methods}')

        print(binned)
        binned.drop(columns=['Time'], inplace=True)
        binned.reset_index(inplace=True)
        binned['Time'] = np.arange(self.data['Time'].values[0], self.data['Time'].values[-1] - interval, interval)
        binned.set_index('Time')

        self.data = binned

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

        trials = Window(data=windows_aligned(source_df=self.data,
                                             fps=self.fps,
                                             alignment_points=indices,
                                             pre_frames=int(pre_period * self.fps),
                                             post_frames=int(post_period * self.fps)),
                        metadata=None,
                        fps=self.fps,
                        indices=indices,
                        trial_type=trial_type)

        if storage_params:
            pass

        return trials

    def event_onsets(self, cols: list) -> dict:
        return event_onsets(self.data, cols=cols)

    def event_intervals(self, cols: list) -> dict:
        return event_intervals(self.data, cols=cols)

    def get_streaks(self, cols: list) -> dict:

        """
        :param cols:    Columns on which to compute streaks


        :return:        Dict of dicts, where streaks[col][event] allows access to a list of streak durations.
        """
        streaks = {}

        ieis = event_intervals(self.data, cols=cols)

        for col, data in ieis.items():
            streaks[col] = streaks_to_lists(streak_df=data)

        return streaks

    def threshold(self, thr: float, binarize: bool = False, columns: List[str] = None, inplace: bool = False):
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

            return data

    def binarize(self, columns: List[str] = None, inplace: bool = False):
        """
        Binarize a dataset. All nonzero entries (including negative numbers!!) will be set to 1.

        :param columns: Columns to binarize.

        :param inplace: If True, updates the self.data parameter to contain only the binarized dataset.

        :returns:           If inplace == False, returns the binarized dataset.
                            If inplace == True, modify inplace and return None
        """
        if inplace:
            self.data[columns][self.data[columns] != 0] = 1
            return None

        else:
            data = self.data.copy(deep=True)
            data[columns][data[columns] != 0] = 1

            return data


if __name__ == '__main__':
    my_series = BehavioralTimeSeries('C://Users\\limogesaw\\Desktop\\mock_data\\Test_4.csv')
    print(my_series.data.shape)
    #my_series.downsample_to_length(method='nearest', length=2000, inplace=True)
    my_series.downsample_by_time(method='mean', interval=1)
    print(my_series.data.shape)

    indices = my_series.data[my_series.data['Shocker on activated'] == 1].index
    print(indices)
    my_trials = my_series.split_by_trial(indices=indices, pre_period=5, post_period=10)
    print(my_trials.indices)
    agg = my_trials.collapse_on(feature='Speed')

    agg = agg.data[::-1]       # This reversal operation should be reserved for plotting

    # Note: ref_range should go into plotting function rather than the collapsed trials object
    # because the highlighted range can be arbitrary and dependent on what the user wants to show
    ref_range = [np.where(agg.T.index.values == 0)[0], np.where(agg.T.index.values == 2)[0]]
    print(agg.shape)
    print(agg.T.index.values)
    #fig = sns.heatmap(agg)
    #print(ref_range)
    #fig.axvspan(ref_range[0], ref_range[1], color='w', alpha=0.3)
    #plt.show()
