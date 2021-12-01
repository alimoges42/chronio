import pandas as pd
import numpy as np
from typing import List
from matplotlib.colors import ListedColormap


class WindowData:
    def __init__(self, windows: List[pd.DataFrame], fps: float, indices: list,
                 trial_type: str = None):
        """
        :param trial_type:
        :param windows: list of DataFrames aligned with reference to some value.
        """
        self.windows = windows
        self.num_windows = len(windows)
        self.fps = fps
        self.indices = indices
        self.trial_type = trial_type

        if all(x.shape == windows[0].shape for x in windows):
            self.dims = windows[0].shape

        else:
            raise ValueError("Each DataFrame must have equal shape.")

    def __str__(self):
        return f'WindowData object containing {self.num_windows} windows of shape {self.dims}'

    def __repr__(self):
        return f'WindowData(num_windows={self.num_windows}, dims={self.dims})'

    def collapse_on(self, column: str) -> np.array:
        """
        :param column: variable from which you would like to extract data
        :return:
        """
        agg_data = np.vstack([window[column].values for window in self.windows])
        return agg_data
        #agg_data = pd.DataFrame(agg_data.T, index=self.windows[0].index).T[::-1]
        #return agg_data

    def spatial_heatmap(self, trial: int = None):
        if trial:
            # Allow specific trial to be plotted
            pass
        pass

    def trajectories_by_trial(self, trial: int = None, cmap: ListedColormap = 'viridis'):
        pass

    def survival_curve(self):
        pass

    def save_windows(self, save_params: dict):

        pass


if __name__ == '__main__':
    from session_data import SessionData

    my_series = SessionData('C://Users\\limogesaw\\Desktop\\mock_data\\Test_4.csv', fps=8)
    my_trials = my_series.split_by_trial(trial_type='so', pre_period=5, post_period=10)

    print(my_trials.windows)
    print(str(my_trials))
    print(repr(my_trials))