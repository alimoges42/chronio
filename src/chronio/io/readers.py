from abc import ABC as _ABC, abstractmethod as _abstractmethod

import pandas as _pd

from chronio.structs import NeuroTimeSeries as _NeuroTimeSeries, \
    BehavioralTimeSeries as _BehavioralTimeSeries, Metadata as _Metadata
from chronio.experiment import stage_from_template


class _Reader(_ABC):
    """
    Base class for reader objects. Dict of params can be passed in for initialization,
    then the reader will load a dataset from the filepath and return a ChronIO-compatible object.
    This affords users the ability to write their own custom reader subclasses so that they can
    retrieve and format datasets to be compatible with the ChronIO interface.
    """

    def __init__(self,
                 params: dict = None):
        self.params = params

    @_abstractmethod
    def load(self,
             fpath: str,
             metadata: _Metadata = None):
        pass


class StageReader(_Reader):
    """
    Used to read a stage from a JSON file.
    Creates a `chronio.experiment.Stage` from the .json file.
    """
    def __init__(self,
                 params: dict = None):
        super().__init__(params=params)

    def load(self,
             fpath: str,
             metadata: _Metadata = None):

        return stage_from_template(fpath)


class AnymazeReader(_Reader):
    """
    Used to read and format a .csv for an individual session that was exported from Anymaze
    Creates a `chronio.structs.BehavioralTimeSeries` from the .csv file.
    """
    def __init__(self,
                 params: dict = None):
        super().__init__(params=params)

    def load(self,
             fpath: str,
             metadata: _Metadata = _Metadata(),
             time_col: str = 'Time',
             encoding: str = None):

        df = _pd.read_csv(fpath, header=0, encoding=encoding)

        return _BehavioralTimeSeries(data=df, time_col=time_col, metadata=metadata)


class IDPSReader(_Reader):
    """
    Used to read and format a .csv for an individual session that was exported from IDPS.
    Creates a `chronio.structs.NeuroTimeSeries` from the .csv file.

    :param params:      Valid parameters are 'include', which may be a string (i.e. 'accepted' or 'rejected')
                        or a list of integers. If a string, the reader will filter the dataframe to include
                        only the cells whose status (located in the second row of the IDPS .csv file)
                        matches that string. If a list of integers, the reader will filter to include only the
                        indices of those cells.
    """
    def __init__(self,
                 params: dict = None):
        super().__init__(params=params)

    def load(self,
             fpath: str,
             metadata: _Metadata = _Metadata()):

        df = _pd.read_csv(fpath, skiprows=[1], header=0)
        abs_time = df[' '].iloc[0]

        df[' '] = df[' '].iloc[:] - abs_time
        df['Time'] = df[' ']
        df = df[df.columns]
        df.drop(columns=[' '], inplace=True)

        cell_status = _pd.read_csv(fpath, nrows=1)
        cell_status.drop(columns=cell_status.columns[0], inplace=True)

        # Filter cells based on string (i.e. 'accepted' or 'rejected')
        if isinstance(self.params['include'], str):
            cell_status = _pd.Series(cell_status.iloc[0], index=cell_status.columns)
            cell_status = cell_status[cell_status.str.contains(self.params['include'])]
            include = cell_status.index.tolist()

        # Filter cells based on list of cell indices
        elif isinstance(self.params['include'], list):
            include = self.params['include']

        else:
            raise ValueError(f'Value of self.params["include"] must be list of integers or a string. \
                             {self.params["include"] = } is invalid.')
        include.insert(0, 'Time')
        df = df[include]

        return _NeuroTimeSeries(data=df, time_col='Time', metadata=metadata)


class PMAT_Reader(_Reader):

    def __init__(self,
                 params: dict = None):
        super().__init__(params=params)

    def load(self,
             fpath: str,
             metadata: _Metadata = _Metadata()):

        df = _pd.read_csv(fpath, header=0)
        df.columns = ['Time', 'DFF']

        return _NeuroTimeSeries(data=df, time_col='Time', metadata=metadata)