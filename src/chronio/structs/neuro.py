#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
neuro.py

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
class NeuroTimeSeries:
    fpath: str