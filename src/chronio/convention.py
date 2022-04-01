#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Design a convention to generate filenames in an understandable and consistent format.
"""

import pathlib
import json
import inspect
from numpy import savetxt

__all__ = ['Convention', 'convention_from_template']


class Convention:
    """
    Holds specifications for a custom filenaming convention.

    Convention objects are passed to other chronio objects including
    :class:`chronio.structs.BehavioralTimeSeries`, :class:`chronio.structs.NeuroTimeSeries`,
    :class:`chronio.structs.Window`, and :class:`chronio.structs.WindowPane`.

    :param directory:       The directory to which to save.
    :type directory:        str

    :param suffix:          The extension of the filename (i.e. csv, xlsx)
    :type suffix:           str

    :param metadata_fields:          Fields of metadata that should be saved.
                            May correspond to column names of the data contained in
                            :class: `chronio.observations.SessionReference` objects
    :type metadata_fields:           list

    :param append_date:     If true, append today's date to the end of the filename. Defaults to False.
    :type append_date:      bool, optional

    :param overwrite:       If true, overwrite a matching file if it is already found in the target directory.
    :type overwrite:        bool, optional

    :param function_kwargs: Arguments to be passed to the Pandas `pd.read_csv()` function
    :type function_kwargs:      dict, optional

    :param savetxt_args:    Arguments to be passed to the Numpy `np.savetxt()` function.
    :type savetxt_args:     dict, optional

    """
    def __init__(self, directory: str, suffix: str, metadata_fields: list,
                 append_date: bool = False, overwrite: bool = False,
                 function_kwargs: dict = {}):
        self.directory = directory
        self.metadata_fields = metadata_fields
        self.suffix = suffix
        self.append_date = append_date
        self.overwrite = overwrite
        self.function_kwargs = function_kwargs

    def get_params(self):
        """
        View the parameters of the convention.

        :return: the attributes of the convention in dict format.
        """
        return self.__dict__

    def export_convention(self, path: str = None):
        """
        Save the convention template to a JSON file.

        :param path: path to save to
        """
        if path:
            path = pathlib.Path(path)
        else:
            path = pathlib.Path(self.directory)
        fname = 'convention.json'

        path = pathlib.PurePath.joinpath(path, fname)

        print(path)

        to_save = self.__dict__

        with open(str(path), 'w') as write_file:
            json.dump(to_save, write_file)


def convention_from_template(template_path: str) -> Convention:
    """
    Load a saved Convention JSON as a Convention object.

    :param template_path:   Path to saved JSON template
    :type template_path:    str

    :return: Convention object restored from the saved file.
    """
    json_data = json.load(open(template_path))
    convention = Convention(directory=json_data['directory'], suffix=json_data['suffix'],
                            metadata_fields=json_data['fields'], append_date=json_data['append_date'],
                            overwrite=json_data['overwrite'], function_kwargs=json_data['function_kwargs'])
    return convention

