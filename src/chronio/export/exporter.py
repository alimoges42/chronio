import datetime as dt
from typing import Any
from pathlib import Path, PurePath
from abc import ABC, abstractmethod

from numpy import ndarray, savetxt
from pandas import DataFrame


class _DataExporter(ABC):
    def __init__(self, obj: Any, metadata: Any, metadata_fields: dict, directory: str, suffix: str,
                 append_date: bool = True, overwrite: bool = False, function_kwargs: dict = {}):
        self.directory = directory
        self._directory = Path(directory)
        self.suffix = suffix

        vals = []
        for attr_key, attr_list in metadata_fields.items():
            for item in attr_list:
                attr_dict = getattr(metadata, attr_key)
                vals.append(str(attr_dict[item]))

        if append_date:
            vals.append(dt.datetime.now().strftime('%m%d%Y'))

        self._fname = f'{"_".join([*vals])}.{suffix}'

        self.fpath = PurePath.joinpath(Path(self.directory), self._fname)

        if overwrite:
            self.export(obj, function_kwargs=function_kwargs)

        else:
            self.prompt(obj, function_kwargs=function_kwargs)

    @abstractmethod
    def export(self, obj: Any, function_kwargs: dict = {}):
        pass

    def prompt(self, obj, function_kwargs: dict = {}):
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
    def __init__(self, obj: DataFrame, metadata: Any, metadata_fields: dict, directory: str, suffix: str,
                 append_date: bool = True, overwrite: bool = False, function_kwargs: dict = {}):
        """
        :param obj:             DataFrame to export

        :param directory:       Directory to save to

        :param suffix:          Suffix (csv)

        :param metadata_fields: Fields to save as part of filename

        :param append_date:     If True, save date as part of filename

        :param overwrite:       If False, will prompt/ask for each filename conflict before overwriting.

        :param function_kwargs: Dict of kwargs passed to either np.savetxt() or pd.to_csv(), depending on
                                user-specified saving convention
        """

        super().__init__(obj=obj,
                         metadata=metadata,
                         metadata_fields=metadata_fields,
                         directory=directory,
                         suffix=suffix,
                         append_date=append_date,
                         overwrite=overwrite,
                         function_kwargs=function_kwargs)

    def export(self, obj: DataFrame, function_kwargs: dict = {}):
        obj.to_csv(self.fpath, **function_kwargs)
        print(f'Object saved as {self.fpath}.')


class _ArrayExporter(_DataExporter):
    def __init__(self, obj: ndarray, metadata: Any, metadata_fields: dict, directory: str, suffix: str,
                 append_date: bool = True, overwrite: bool = False, function_kwargs: dict = {}):
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

        super().__init__(obj=obj,
                         metadata=metadata,
                         metadata_fields=metadata_fields,
                         directory=directory,
                         append_date=append_date,
                         suffix=suffix,
                         overwrite=overwrite,
                         function_kwargs=function_kwargs)

    def export(self, obj: ndarray, function_kwargs: dict = {}):
        savetxt(self.fpath, obj, **function_kwargs)
        print(f'Object saved as {self.fpath}.')

