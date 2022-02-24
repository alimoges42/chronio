from abc import ABC
from typing import Any
from chronio.export.exporter import _DataExporter
from .metadata import Metadata


class _DataStructure(ABC):
    def __init__(self, data: Any, metadata: Metadata):
        self.data = data
        self.meta = metadata

    def export(self, exporter_kwargs: dict = None):
        exporter = _DataExporter(obj=self, **exporter_kwargs)
        exporter.export(self.data)


if __name__ == '__main__':
    test = _DataStructure()
    test.export({'obj': '/home/aaron/', 'fields': [1, 2, 3],
                 'overwrite': False})