import datetime as dt
import glob
from pathlib import Path
import pandas as pd


today = dt.datetime.now()
print(today.strftime('%m%d%Y'))


class DataExporter:
    def __init__(self, directory: Path, suffix: str, fields: list,
                 append_date: bool = True, overwrite: bool = False) -> None:

        self.directory = directory
        self.fields = fields
        self.suffix = suffix
        self._fname = f'{"_".join([*fields])}.{suffix}'
        self.fpath = Path(f'{self.directory}{self._fname}')

        if append_date:
            self.date = dt.datetime.now().strftime('%m%d%Y')

        if overwrite:
            self.export()

        else:
            self.prompt()


    def export(self):
        pass

    def prompt(self):
        if self.fpath.is_file():
            answer = input(f'File {self._fname} already exists in {self.directory}. Overwrite (Y/N)?')
            if answer in ['N', 'n', 'No', 'no']:
                print('File not saved.')

            elif answer in ['Y', 'y', 'Yes', 'yes']:
                self.export()
                print(f'File saved to {self.fpath}')

            else:
                print('Invalid input. Please try again.')
                self.prompt()


if __name__ == '__main__':
    fields_ = ['Aaron', 'Limoges']
    directory_ = '/home/aaron/'
    suffix_ = 'csv'

    d = DataExporter(directory=directory_,
                     suffix=suffix_,
                     fields=fields_)

    print(d.fpath)