#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time Series

This submodule contains classes for working with time series data.
It is useful for storing the raw time series dataset as well as its metadata.

@author: Aaron Limoges
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from anymaze.behavior_utils.structs.window_data import WindowData
from anymaze.process.slicingtools import windows_aligned
from anymaze.process.analyses import event_onsets, event_intervals, streaks_to_lists


@dataclass
class BehavioralTimeSeries:
    fpath: str

    def __post_init__(self):
        self.df = pd.read_csv(self.fpath)
        self.fps = self.df.shape[0] / round(self.df['Time'].values[-1], 0)

    def split_by_trial(self,
                       indices: list,
                       trial_type: str = None,
                       pre_period: float = 0,
                       post_period: float = 0,
                       storage_params: str = None) -> WindowData:

        """
        :param indices:         Indices to align to

        :param trial_type:      User-defined name of trial type

        :param pre_period:      Time (in seconds) desired to obtain prior to trial onset

        :param post_period:     Time (in seconds) desired to obtain following end of trial

        :param storage_params:  Dict of parameters for saving

        :return:                List of aligned trial data
        """

        trials = WindowData(
                            data=windows_aligned(source_df=self.df,
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
        return event_onsets(self.df, cols=cols)

    def event_intervals(self, cols: list) -> dict:
        return event_intervals(self.df, cols=cols)

    def get_streaks(self, cols: list) -> dict:

        """
        :param cols:    Columns on which to compute streaks


        :return:        Dict of dicts, where streaks[col][event] allows access to a list of streak durations.
        """
        streaks = {}

        ieis = event_intervals(self.df, cols=cols)

        for col, data in ieis.items():
            streaks[col] = streaks_to_lists(streak_df=data)

        return streaks


if __name__ == '__main__':
    my_series = BehavioralTimeSeries('C://Users\\limogesaw\\Desktop\\mock_data\\Test_4.csv')
    indices = my_series.df[my_series.df['Shocker on activated'] == 1].index
    print(indices)
    my_trials = my_series.split_by_trial(indices=indices, pre_period=5, post_period=10)
    print(my_trials.indices)
    agg = my_trials.collapse_on(feature='Speed')
    print(agg.mean())

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
