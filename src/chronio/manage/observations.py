from __future__ import annotations
from typing import List
from pathlib import Path, PurePath
import pandas as pd
from chronio.structs.raw_structs import BehavioralTimeSeries, NeuroTimeSeries
#from chronio.structs.neuro import NeuroTimeSeries


class SessionReference:
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
        :param col_args:    dict whose keys are column names,
                            and values are lists of target values for that column

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
        :param row:         a pd.Series where one or more entries defines a path to a time series csv

        :param mappings:    a dict that maps index names of the row parameter to a supported data structure such as
                            BehavioralTimeSeries or NeuroTimeSeries
        """
        self.data = row
        self.behavior_cols = []
        self.neuro_cols = []
        self._mappings = mappings

        for key, value in mappings.items():
            if value == BehavioralTimeSeries:
                self.behavior_cols.append(key)

            elif value == NeuroTimeSeries:
                self.neuro_cols.append(key)

        # Assume all remaining columns constitute some form of metadata
        _nonmeta_cols = [*self.behavior_cols, *self.neuro_cols]

        self.meta_cols = [idx for idx in self.data.index if idx not in _nonmeta_cols]
        self.meta = row[self.meta_cols]

    def load(self, subset: List[str] = []):

        load_cols = [[*self.behavior_cols, *self.neuro_cols], subset]
        load_cols = set().union(*load_cols)

        attr_names = [col.replace(' ', '_') for col in load_cols]

        for col, attr_name in zip(load_cols, attr_names):
            print(col, attr_name)
            path_to_data = Path(self.data[col])
            setattr(self, attr_name, self._mappings[col](str(path_to_data)))


if __name__ == '__main__':
    f = 'C:\\Users\\limogesaw\\Desktop\\Test_Project\\Test_Experiment\\Test_Cohort1\\Day1\\Mouse1\\Dyn-Cre_CeA_Photometry - Test 42.csv'
    row = pd.Series({'name': 'Aaron', 'behavior file': f, 'ID': '5'})

    obs = Session(row=row, mappings={'behavior file': BehavioralTimeSeries})
    obs.load()

    print(obs.behavior_file)