from typing import Any as _Any, Callable as _Callable, List as _List, Dict as _Dict
from pathlib import Path, PurePath
from abc import ABC, abstractmethod

import pandas as _pd
from numpy import ndarray


class Reader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load(self):
        pass


def read_IDPS(fpath: str,
              round_time: int = None,
              include: _List[str] = 'accepted'):
    df = _pd.read_csv(fpath, skiprows=[1], header=0)
    abs_time = df[' '].iloc[0]

    df[' '] = df[' '].iloc[:] - abs_time
    if round_time:
        df['Time'] = df[' '].round(round_time)
    else:
        df['Time'] = df[' ']
    df = df[df.columns]
    df.drop(columns=[' '], inplace=True)

    cell_status = _pd.read_csv(fpath, nrows=1)
    if isinstance(include, str):
        include = [include]

    bool_status = None
    for label in include:
        if bool_status is None:
            bool_status = cell_status.iloc[0].str.contains(label)
            continue
        else:
            bool_status += cell_status.iloc[0].str.contains(label)
    bool_status = bool_status.astype(bool)

    accepted_cols = bool_status[bool_status == True].index.tolist()
    accepted_cols.insert(0, 'Time')
    df = df[accepted_cols]
    return df
