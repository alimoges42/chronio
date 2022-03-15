import datetime as dt
from typing import Any
from pathlib import Path, PurePath
from abc import ABC, abstractmethod

from numpy import ndarray, savetxt
from pandas import DataFrame


class _DataExporter(ABC):
    def __init__(self, obj: Any, directory: str, suffix: str, fields: list,
                 append_date: bool = True, overwrite: bool = False, function_kwargs: dict = None):
        self.directory = directory
        self._directory = Path(directory)
        self.fields = fields
        self.suffix = suffix
        self._fname = f'{"_".join([*fields])}.{suffix}'

        self.fpath = PurePath.joinpath(Path(self.directory), self._fname)

        if append_date:
            self.date = dt.datetime.now().strftime('%m%d%Y')

        if overwrite:
            self.export(obj, function_kwargs=function_kwargs)

        else:
            self.prompt(obj, function_kwargs=function_kwargs)

    @abstractmethod
    def export(self, obj: Any, function_kwargs: dict = None):
        pass

    def prompt(self, obj, function_kwargs: dict = None):
        if self.fpath.exists():
            answer = input(f'File {self._fname} already exists in {self.directory}. Overwrite (Y/N)?')
            if answer in ['N', 'n', 'No', 'no']:
                print('File not saved.')

            elif answer in ['Y', 'y', 'Yes', 'yes']:
                self.export(obj, function_kwargs=function_kwargs)
                print(f'File saved to {self.fpath}')

            else:
                print('Invalid input. Please try again.')
                self.prompt()

        else:
            self.export(obj=obj, function_kwargs=function_kwargs)


class _DataFrameExporter(_DataExporter):
    def __init__(self, obj: DataFrame, directory: str, suffix: str, fields: list,
                 append_date: bool = True, overwrite: bool = False, function_kwargs: dict = None):
        """
        :param obj:             DataFrame to export

        :param directory:       Directory to save to

        :param suffix:          Suffix (csv)

        :param fields:          Fields to save as part of filename

        :param append_date:     If True, save date as part of filename

        :param overwrite:       If False, will prompt/ask for each filename conflict before overwriting.

        :param function_kwargs: Dict of kwargs passed to either np.savetxt() or pd.to_csv(), depending on
                                user-specified saving convention
        """

        super().__init__(obj, directory, suffix, fields, append_date, overwrite, function_kwargs=function_kwargs)

    def export(self, obj: DataFrame, function_kwargs: dict = None):
        obj.to_csv(self.fpath, **function_kwargs)
        print(f'Object saved as {self.fpath}.')


class _ArrayExporter(_DataExporter):
    def __init__(self, obj: ndarray, directory: str, suffix: str, fields: list,
                 append_date: bool = True, overwrite: bool = False, function_kwargs: dict = None):
        """
        :param obj:             array to export

        :param directory:       Directory to save to

        :param suffix:          Suffix (csv)

        :param fields:          Fields to save as part of filename

        :param append_date:     If True, save date as part of filename

        :param overwrite:       If False, will prompt/ask for each filename conflict before overwriting.

        :param function_kwargs: Dict of kwargs passed to either np.savetxt() or pd.to_csv(), depending on
                                user-specified saving convention
        """

        super().__init__(obj, directory, suffix, fields, append_date, overwrite, function_kwargs=function_kwargs)

    def export(self, obj: ndarray, function_kwargs: dict = None):
        savetxt(self.fpath, obj, **function_kwargs)
        print(f'Object saved as {self.fpath}.')


if __name__ == '__main__':
    fields_ = ['Aaron', 'Limoges']
    directory_ = 'C://Users/limogesaw/Desktop'
    suffix_ = 'csv'
    data = {'Names': ['Aaron', 'Bob', 'Jackie'],
           'ages': [30, 24, 57]}

    d = _DataFrameExporter(obj=DataFrame(data=data),
                           directory=directory_,
                           suffix=suffix_,
                           fields=fields_, function_kwargs={})
    print(d.fpath)

    a = _ArrayExporter(obj=ndarray([3, 3]),
                       directory=directory_,
                       suffix=suffix_,
                       fields=fields_, function_kwargs={})


