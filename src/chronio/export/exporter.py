import datetime as dt
from typing import Any, Callable
from pathlib import Path
from abc import ABC
import pandas as pd

today = dt.datetime.now()


class _DataExporter(ABC):
    def __init__(self, obj: Any, directory: str, suffix: str, fields: list,
                 append_date: bool = True, overwrite: bool = False):

        self.directory = directory
        self._directory = Path(directory)
        self.fields = fields
        self.suffix = suffix
        self._fname = f'{"_".join([*fields])}.{suffix}'

        self.fpath = Path(f'{self.directory}{self._fname}')

        if append_date:
            self.date = dt.datetime.now().strftime('%m%d%Y')

        if overwrite:
            self.export(obj)

        else:
            self.prompt()

    def export(self, obj: Any, function_kwargs: dict = None):
        pass

    def prompt(self):
        if self.fpath.exists():
            answer = input(f'File {self._fname} already exists in {self.directory}. Overwrite (Y/N)?')
            if answer in ['N', 'n', 'No', 'no']:
                print('File not saved.')

            elif answer in ['Y', 'y', 'Yes', 'yes']:
                self.export()
                print(f'File saved to {self.fpath}')

            else:
                print('Invalid input. Please try again.')
                self.prompt()


class _DataFrameExporter(_DataExporter):
    def __init__(self, obj: pd.DataFrame, directory: str, suffix: str, fields: list,
                 append_date: bool = True, overwrite: bool = False):
        """
        :param obj:             DataFrame to export

        :param directory:       Directory to save to

        :param suffix:          Suffix (csv)

        :param fields:          Fields to save as part of filename

        :param append_date:     If true, save date as part of filename

        :param overwrite:       If false, will prompt/ask for each filename conflict before overwriting.
        """

        super().__init__(obj, directory, suffix, fields, append_date, overwrite)

    def export(self, obj: Any, function_kwargs: dict = None):
        obj.to_csv(self.fpath, **function_kwargs)
        print(f'Object saved as {self.fpath}.')


if __name__ == '__main__':
    fields_ = ['Aaron', 'Limoges']
    directory_ = '/home/aaron/'
    suffix_ = 'csv'

    d = _DataExporter(directory=directory_,
                      suffix=suffix_,
                      fields=fields_)
    print(d.fpath)
