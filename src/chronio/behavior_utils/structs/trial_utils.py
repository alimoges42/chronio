#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trial Utils

This contains a SessionReference class, which reads from the "tests.htm" file produced by Anymaze
to allow the user to rapidly filter the universe of tests to select desired trial types for desired replicates.

@author: Aaron Limoges
"""

from pathlib import Path, PurePath
import pandas as pd


class SessionReference:

    def __init__(self, fpath):
        self.fpath = fpath
        self._fpath = Path(self.fpath)
        self._basedir = self._fpath.parent
        self.basedir = str(self._fpath.parent)

        if PurePath(self.fpath).suffix == 'csv':
            self.data = pd.read_csv(fpath)

        elif PurePath(self.fpath).suffix == 'xlsx':
            self.data = pd.read_excel(fpath)

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

    ref = SessionReference(f)
    print(ref.data.columns)
    print(ref.filter({'Test': [2, 3, 6, 10], 'Animal': 2}))
    print(ref.stages, ref.mice)

