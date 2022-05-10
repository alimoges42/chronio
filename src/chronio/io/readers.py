from typing import Any, Callable
from pathlib import Path, PurePath
from abc import ABC, abstractmethod

from pandas import DataFrame
from numpy import ndarray


class _DataReader(ABC):
    def __init__(self, fpath):
        pass


class _IDPSReader(_DataReader):
    def __init__(self, fpath):
        pass

