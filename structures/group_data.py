#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Group Data

This contains classes for working with grouped datasets, i.e. similar datasets across multiple replicates.

@author: Aaron Limoges
"""

import pandas as pd
import numpy as np


class RMGroup:
    """
    Repeated Measures Group data.

    Works with datasets whose rows represent samples and
    whose columns represent repeated measures (i.e. trials or sessions) of a specific feature.

    A subset of columns (id_cols) may also be used to supply identifier features
    associated with each sample (i.e. groups)

    Sample IDs are assumed to be in the index of the dataframe
    """

    def __init__(self, data: pd.DataFrame, id_cols: list = None):
        self.data = data
        self.samples = data.index.values
        self.id_cols = id_cols


class Group:
    def __init__(self):
        pass
    pass
