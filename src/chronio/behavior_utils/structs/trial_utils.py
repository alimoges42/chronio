#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trial Utils

This contains a TrialReference class, which reads from the "tests.htm" file produced by Anymaze
to allow the user to rapidly filter the universe of tests to select desired trial types for desired replicates.

@author: Aaron Limoges
"""

import pathlib
import pandas as pd
from dataclasses import dataclass


@dataclass
class TrialReference:
    fpath: str

    def __post_init__(self):
        file_ = pd.read_html(self.fpath)
        _data = pd.DataFrame()

        for row in file_[1:]:
            _data = _data.append(row)

        _data.drop(columns=_data.columns[0], inplace=True)

        _data = pd.DataFrame(data=_data.iloc[1:].values, columns=_data.iloc[0])
        self.data = _data
        self.mice = self.data['Animal'].unique()
        self.stages = self.data['Stage'].unique()
        self.statuses = self.data['Testing status'].unique()
        self._fpath = pathlib.Path(self.fpath)
        self._basedir = self._fpath.parent
        self.basedir = str(self._fpath.parent)

    def filter(self, col_args: dict) -> pd.DataFrame:
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

        return df


if __name__ == '__main__':
    lab_folder = '//nih.gov\\nimhfileshare\\Lab\\UNSI\\NIMH DIRP NSI'
    user = 'Aaron'
    experiment = 'Dyn-deletion_CeA'
    subdirs = 'Cohort1\\Behavior'

    htm = 'Tests.htm'

    f = '\\'.join([lab_folder, user, experiment, subdirs, htm])
    print(f)

    ref = TrialReference(f)
    print(ref.data.columns)
    print(ref.filter({'Test': [2, 3, 6, 10], 'Animal': 2}))
    print(ref.stages, ref.mice)

