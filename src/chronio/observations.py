#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for managing and mapping files associated with multiple specimens.
"""

from __future__ import annotations
from typing import List
import pathlib
import pandas as pd

from chronio.structs import BehavioralTimeSeries, NeuroTimeSeries, Metadata
from chronio.experiment import Stage, stage_from_template

__all__ = ['SessionReference', 'Session']


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

        if not self.data:
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

        return SessionReference(fpath=self.fpath, reference_data=df)


class Session:
    def __init__(self, row: pd.Series, mappings: dict, stage_dir: str = None):

        """
        Built to hold multiple time series datasets as well as metadata about a given recorded session.
        Instances of this object will have attributes corresponding to index names of the provided pd.Series.

        :param row:         a pd.Series where one or more entries defines a path to a time series csv
        :type row:          pd.Series

        :param mappings:    a dict that maps index names of the row parameter to a supported data structure such as
                            :class:`chronio.structs.raw_structs.BehavioralTimeSeries` or NeuroTimeSeries
        :type mappings:     dict

        :param stage_dir:   path to a directory that holds the JSON corresponding to the stage_name for this session
        :type stage_dir:    str
        """

        self.data = row
        self.behavior_cols = []
        self.neuro_cols = []
        self.stage_dir = stage_dir
        self.stage_name = None
        self.stage = None

        self.mappings = mappings

        for key, value in mappings.items():
            if value == BehavioralTimeSeries:
                self.behavior_cols.append(key)
                print(f'BehavioralTimeSeries mapped to column "{key}".')

            elif value == NeuroTimeSeries:
                self.neuro_cols.append(key)
                print(f'NeuroTimeSeries mapped to column "{key}".')

            elif value == Stage:
                self.stage_name = row[key]
                fpath = pathlib.Path(stage_dir)
                stage_fpath = pathlib.Path.joinpath(fpath, f'{self.stage_name}.json')
                self.stage = stage_from_template(str(stage_fpath))
                print(f'Stage mapped to column "{key}".')

        # Assume all remaining columns constitute some form of metadata.
        # Stage column is also considered metadata.
        _nonmeta_cols = [*self.behavior_cols, *self.neuro_cols]

        self.meta_cols = [idx for idx in self.data.index if idx not in _nonmeta_cols]
        self.meta = row[self.meta_cols]
        self.meta.to_dict()

    def load(self, subset: List[str] = []):
        """
        Load all or a subset of files associated with this object.

        :param subset:  desired index names to load
        :type subset:   List[str]
        """

        cols_to_load = [[*self.behavior_cols, *self.neuro_cols], subset]
        cols_to_load = set().union(*cols_to_load)

        attr_names = [col.replace(' ', '_') for col in cols_to_load]

        for col, attr_name in zip(cols_to_load, attr_names):
            fpath = str(pathlib.Path(self.data[col]))

            # TODO: map other cols to metadata
            if self.stage_name:
                metadata = Metadata(fpath=fpath,
                                    stage=self.stage_name)
                metadata.set_val('session', self.meta)
            else:
                metadata = Metadata(fpath=fpath)
                metadata.set_val('session', self.meta)

            setattr(self, attr_name, self.mappings[col](fpath=fpath,
                                                        metadata=metadata))

            print(f'Data from column "{col}" successfully loaded as self.{attr_name}.')
