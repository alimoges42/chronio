#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for managing and mapping files associated with multiple specimens.
"""

from __future__ import annotations
from typing import List as _List, Dict as _Dict
import pathlib
import pandas as pd

import chronio.io.readers
from chronio.structs import _TimeSeries, Metadata

__all__ = ['SessionReference', 'Session', 'session_from_row']


class SessionReference:
    """
    A class which is used to hold and access metadata about an experiment. This can be constructed
    by providing a filepath to a csv or xlsx file that contains this information, or alternatively can
    be built from a DataFrame that already exists in memory. At least one column of this should contain filepaths
    to csv or xlsx files that contain time series data.

    :param fpath:           Path to the reference file.
    :type fpath:            str

    :param reference_data:  An existing DataFrame with metadata on experiments.
    :type reference_data:   pd.DataFrame
    """
    def __init__(self, fpath: str = None, reference_data: pd.DataFrame = None):
        self.fpath = fpath
        self.data = reference_data

        if self.data is None:
            if self.fpath:
                if pathlib.PurePath(self.fpath).suffix == '.csv':
                    self.data = pd.read_csv(fpath)

                elif pathlib.PurePath(self.fpath).suffix == '.xlsx':
                    self.data = pd.read_excel(fpath)

                else:
                    raise ValueError('Unsupported extension. Supported extensions are .csv and .xlsx.')

    def filter(self, col_args: dict) -> SessionReference:
        """
        Filter data to select only a certain subset to analyze.

        :param col_args:    dict whose keys are column names,
                            and values are lists of target values for that column
        :type col_args:     dict

        :return:            a filtered dataframe containing only the selected columns
        """

        df = self.data.copy()

        print(col_args)
        for col, val in col_args.items():
            if isinstance(val, list):
                df = df[df[col].isin(val)]
            else:
                df = df[df[col] == val]

        return self.__class__(fpath=self.fpath, reference_data=df)


class Session:
    def __init__(self, attrs: _Dict[str, _TimeSeries], meta: Metadata = Metadata()):
        for attr_name, dataset in attrs.items():
            setattr(self, attr_name, dataset)
        self.meta = meta


def session_from_row(row: pd.Series,
                      mappings: _Dict[str, chronio.io.readers._Reader],
                      subset: _List[str] = None,
                      meta_cols: _List[str] = None) -> Session:
    """
    :param row:         Typically, this is a row from the SessionReference object.
    :type row:          pd.Series

    :param mappings:    dict that assigns at least one row index to a :class: `chronio.structs.BehavioralTimeSeries` or
                        :class: `chronio.structs.NeuroTimeSeries` object.
                        Indices not mapped are assumed to hold metadata.
    :type mappings:     dict

    :param subset:      If desired, load a subset of columns
    :type subset:       List[str]

    :return:
    """

    if not subset:
        subset = []

    stage_name = None
    stage = None

    # Assume all remaining columns constitute some form of metadata.
    # Stage column is also considered metadata.
    _nonmeta_cols = [i for i in mappings.keys() if i != 'stage']

    if meta_cols is None:
        meta_cols = [idx for idx in row.index if idx not in _nonmeta_cols]

    meta = row[meta_cols].to_dict()

    if stage:
        setattr(meta, 'stage', stage)

    cols_to_load = [_nonmeta_cols, subset]
    cols_to_load = set().union(*cols_to_load)

    attr_names = [col.replace(' ', '_') for col in cols_to_load]
    attrs = {}

    metadata = None

    for col, attr_name in zip(cols_to_load, attr_names):
        fpath = str(pathlib.Path(row[col]))

        if stage_name:
            metadata = Metadata(fpath=fpath, stage_name=stage_name)
            metadata.set_val('session', meta)
        else:
            metadata = Metadata(fpath=fpath)
            metadata.set_val('session', meta)
        print(f'Data from column "{col}" successfully loaded as self.{attr_name}.')

        #if isinstance(mappings[col], chronio.Stage):
        #    continue
        #else:
        #    obj = mappings[col]
        #    attrs[attr_name] = obj.load(fpath=fpath)

        obj = mappings[col]
        attrs[attr_name] = obj.load(fpath=fpath, metadata=metadata)

    session = Session(attrs=attrs, meta=metadata)

    return session
