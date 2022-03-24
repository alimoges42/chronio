from __future__ import annotations
from typing import List
from pathlib import Path, PurePath
import pandas as pd
from chronio.structs.raw_structs import BehavioralTimeSeries, NeuroTimeSeries


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
                if PurePath(self.fpath).suffix == '.csv':
                    self.data = pd.read_csv(fpath)

                elif PurePath(self.fpath).suffix == '.xlsx':
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
    def __init__(self, row: pd.Series, mappings: dict = None):

        """
        Built to hold multiple time series datasets as well as metadata about a given recorded session.
        Instances of this object will have attributes corresponding to index names of the provided pd.Series.

        :param row:         a pd.Series where one or more entries defines a path to a time series csv
        :type row:          pd.Series

        :param mappings:    a dict that maps index names of the row parameter to a supported data structure such as
                            :class:`chronio.structs.raw_structs.BehavioralTimeSeries` or NeuroTimeSeries
        :type mappings:     dict
        """
        self.data = row
        self.behavior_cols = []
        self.neuro_cols = []
        self._mappings = mappings

        for key, value in mappings.items():
            if value == BehavioralTimeSeries:
                self.behavior_cols.append(key)
                print(f'BehavioralTimeSeries mapped to column "{key}".')

            elif value == NeuroTimeSeries:
                self.neuro_cols.append(key)
                print(f'NeuroTimeSeries mapped to column "{key}".')

        # Assume all remaining columns constitute some form of metadata
        _nonmeta_cols = [*self.behavior_cols, *self.neuro_cols]

        self.meta_cols = [idx for idx in self.data.index if idx not in _nonmeta_cols]
        self.meta = row[self.meta_cols]

    def load(self, subset: List[str] = []):
        """
        Load all or a subset of files associated with this object.

        :param subset:  desired index names to load
        :type subset:   List[str]
        """

        load_cols = [[*self.behavior_cols, *self.neuro_cols], subset]
        load_cols = set().union(*load_cols)

        attr_names = [col.replace(' ', '_') for col in load_cols]

        for col, attr_name in zip(load_cols, attr_names):
            path_to_data = Path(self.data[col])
            setattr(self, attr_name, self._mappings[col](fpath=str(path_to_data)))
            print(f'Data from column "{col}" successfully loaded as self.{attr_name}.')
