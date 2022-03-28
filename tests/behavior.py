#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
behavior.py

This submodule contains classes for working with behavioral time series data.
It is useful for storing the raw time series dataset as well as its metadata.

@author: Aaron Limoges
"""

from typing import Any
import pandas as pd
import numpy as np

from chronio.structs.structs import BehavioralTimeSeries
#from chronio.structs.windows import Window
from chronio.process.analyses import windows_aligned





if __name__ == '__main__':
    my_series = BehavioralTimeSeries(fpath='C://Users\\limogesaw\\Desktop\\mock_data\\Test_4.csv')
    print(my_series.data.shape)
    #my_series.downsample_to_length(method='nearest', length=2000, inplace=True)
    my_series.downsample_by_time(method='mean', interval=1)
    print(my_series.data.shape)

    indices = my_series.data[my_series.data['Shocker on activated'] == 1].index
    print(indices)
    my_trials = my_series.split_by_trial(indices=indices, pre_period=5, post_period=10)
    print(my_trials.metadata.window_params['indices'])
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
