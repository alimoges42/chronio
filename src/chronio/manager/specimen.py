from typing import Any
import pandas as pd


class Specimen:
    def __init__(self, **files):
        for key in files:
            print(key)
            setattr(self, key, files[key])


if __name__ == '__main__':
    spec = Specimen(file1={'path': 'yo', 'obj': 'behavior'}, file2={'path': 'hey', 'obj': 'neuro'})
    print(spec.file2)