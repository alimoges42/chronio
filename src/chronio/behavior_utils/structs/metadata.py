from typing import Any
from abc import ABC
from pathlib import Path


class Metadata(ABC):
    def __init__(self, kwargs: dict = None):
        for key in kwargs:
            setattr(self, key, kwargs[key])

        if self.path:
            self._path = Path(self.path)
