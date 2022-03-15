from abc import ABC
from typing import Any
from numpy import ndarray
from pandas import DataFrame

from chronio.designer.convention import Convention
from chronio.export.exporter import _DataFrameExporter, _ArrayExporter


class _DataStructure(ABC):
    def __init__(self, data: Any, metadata: Any = None):
        self.data = data
        self.meta = metadata

    def export(self, convention: Convention = None, function_kwargs: dict = None, **exporter_kwargs):
        """
        Parameters supplied by exporter_kwargs will replace those supplied by the convention object. This is to
        allow users on-the-fly editing without having to specify all the new fields of a convention object if they
        want to make a minor change to the convention.
        """

        # Overwrites convention params with user-specified export_kwargs (if supplied)
        export_kwargs = convention.get_params()
        export_kwargs.update(exporter_kwargs)

        if type(self.data) == ndarray:
            exporter = _ArrayExporter(obj=self.data, **export_kwargs)

        elif type(self.data) == DataFrame:
            # TODO: Support both CSV and XLSX export options - could be achieved by editing _DataFrameExporter?
            exporter = _DataFrameExporter(obj=self.data, **export_kwargs)

        else:
            raise ValueError(f'No export protocol for {type(self.data)} exists.')

        if function_kwargs:
            exporter.export(self.data, **function_kwargs)

        else:
            exporter.export(self.data)


if __name__ == '__main__':
    test = _DataStructure(data=DataFrame())
    test.export(convention=Convention(directory='//', suffix='csv', append_date=True, overwrite=False, fields=['name']),
                directory='//aaron/')

    test.export(function_kwargs={'header': False}, fields=['name', 'date'], overwrite=False, directory='/home/aaron',
                suffix='csv')
