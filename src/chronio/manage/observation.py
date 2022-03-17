from typing import List
from pathlib import Path
import pandas as pd
from chronio.structs.behavior import BehavioralTimeSeries
from chronio.structs.neuro import NeuroTimeSeries


class Observation:
    def __init__(self, row: pd.Series, mappings: dict = None):

        self.data = row
        self.behavior_cols = []
        self.neuro_cols = []

        for key, value in mappings.items():
            if value == BehavioralTimeSeries:
                self.behavior_cols.append(key)

            elif value == NeuroTimeSeries:
                self.neuro_cols.append(key)

        # Assume all remaining columns constitute some form of metadata
        _nonmeta_cols = [*self.behavior_cols, *self.neuro_cols]

        self.meta_cols = [idx for idx in self.data.index if idx not in _nonmeta_cols]
        self.meta = row[self.meta_cols]

    def load(self, subset: List[str] = []):

        if self.behavior_cols:
            _behavior_cols = [col.replace(' ', '_') for col in self.behavior_cols]

        if self.neuro_cols:
            _neuro_cols = [col.replace(' ', '_') for col in self.neuro_cols]

        for col in _behavior_cols:
            path_to_data = Path(self.data[col])
            setattr(self, col, BehavioralTimeSeries(str(path_to_data)))

        for col in _neuro_cols:
            path_to_data = Path(self.data[col])
            setattr(self, col, NeuroTimeSeries(str(path_to_data)))

if __name__ == '__main__':
    f = 'C:\\Users\\limogesaw\\Desktop\\Test_Project\\Test_Experiment\\Test_Cohort1\\Day1\\Mouse1\\Dyn-Cre_CeA_Photometry - Test 42.csv'
    row = pd.Series({'name': 'Aaron', 'behavior': f, 'ID': '5'})

    obs = Observation(row=row, mappings={'behavior': BehavioralTimeSeries})
    obs.load()

    print(obs.behavior)