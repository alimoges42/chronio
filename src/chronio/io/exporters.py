import datetime as dt
from typing import Any
from pathlib import Path, PurePath
from abc import ABC, abstractmethod

from numpy import ndarray, savetxt
from pandas import DataFrame


class _DataExporter(ABC):
    def __init__(self, obj: Any, obj_type: str, metadata: Any, metadata_fields: dict, directory: str, suffix: str,
                 append_date: bool = True, overwrite: bool = False, function_kwargs: dict = None):

        if not function_kwargs:
            function_kwargs = {}

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

        vals.append(obj_type)

        self._fname = f'{"_".join([*vals])}.{suffix}'

        self.fpath = PurePath.joinpath(Path(self.directory), self._fname)

        if overwrite:
            self.export(obj, function_kwargs=function_kwargs)

        else:
            self.prompt(obj, function_kwargs=function_kwargs)

    @abstractmethod
    def export(self, obj: Any, function_kwargs: dict = None):
        pass

    def prompt(self, obj, function_kwargs: dict = None):
        if not function_kwargs:
            function_kwargs = {}

        if self.fpath.exists():
            answer = input(f'File {self._fname} already exists in {self.directory}. Overwrite (Y/N)?')
            if answer in ['N', 'n', 'No', 'no']:
                print('File not saved.')

            elif answer in ['Y', 'y', 'Yes', 'yes']:
                self.export(obj=obj, function_kwargs=function_kwargs)
                print(f'File saved to {self.fpath}')

            else:
                print('Invalid input. Please try again.')
                self.prompt(obj=obj, function_kwargs=function_kwargs)

        else:
            self.export(obj=obj, function_kwargs=function_kwargs)


class _DataFrameExporter(_DataExporter):
    def __init__(self, obj: DataFrame, obj_type: str, metadata: Any, metadata_fields: dict, directory: str, suffix: str,
                 append_date: bool = False, overwrite: bool = False, function_kwargs: dict = None):
        """
        :param obj:             DataFrame to io
        :type obj:              pd.DataFrame

        :param metadata:        Metadata to io
        :type metadata:         Any

        :param metadata_fields: Fields to save as part of filename
        :type metadata_fields:  dict

        :param directory:       Directory to save to
        :type directory:        str

        :param suffix:          Suffix (csv)
        :type suffix:           str

        :param append_date:     If True, save date as part of filename (False by default)
        :type append_date:      bool

        :param overwrite:       If False, will prompt/ask for each filename conflict before overwriting.
        :type overwrite:        bool

        :param function_kwargs: Dict of kwargs passed to either np.savetxt() or pd.to_csv(), depending on
                                user-specified saving convention
        :type function_kwargs:  dict
        """
        if not function_kwargs:
            function_kwargs = {}

        super().__init__(obj=obj,
                         metadata=metadata,
                         obj_type=obj_type,
                         metadata_fields=metadata_fields,
                         directory=directory,
                         suffix=suffix,
                         append_date=append_date,
                         overwrite=overwrite,
                         function_kwargs=function_kwargs)

    def export(self, obj: DataFrame, function_kwargs: dict = None):
        if not function_kwargs:
            function_kwargs = {}
        obj.to_csv(self.fpath, **function_kwargs)
        print(f'Object saved as {self.fpath}.')


class _ArrayExporter(_DataExporter):
    def __init__(self, obj: ndarray, metadata: Any, obj_type: str, metadata_fields: dict, directory: str, suffix: str,
                 append_date: bool = True, overwrite: bool = False, function_kwargs: dict = None):
        """
        :param obj:             DataFrame to io
        :type obj:              pd.DataFrame

        :param metadata:        Metadata to io
        :type metadata:         Any

        :param metadata_fields: Fields to save as part of filename
        :type metadata_fields:  dict

        :param directory:       Directory to save to
        :type directory:        str

        :param suffix:          Suffix (csv)
        :type suffix:           str

        :param append_date:     If True, save date as part of filename (False by default)
        :type append_date:      bool

        :param overwrite:       If False, will prompt/ask for each filename conflict before overwriting.
        :type overwrite:        bool

        :param function_kwargs: Dict of kwargs passed to either np.savetxt() or pd.to_csv(), depending on
                                user-specified saving convention
        :type function_kwargs:  dict
        """
        if not function_kwargs:
            function_kwargs = {}

        super().__init__(obj=obj,
                         metadata=metadata,
                         obj_type=obj_type,
                         metadata_fields=metadata_fields,
                         directory=directory,
                         append_date=append_date,
                         suffix=suffix,
                         overwrite=overwrite,
                         function_kwargs=function_kwargs)

    def export(self, obj: ndarray, function_kwargs: dict = None):
        if not function_kwargs:
            function_kwargs = {}
        savetxt(self.fpath, obj, **function_kwargs)
        print(f'Object saved as {self.fpath}.')

