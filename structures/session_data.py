import pandas as pd
import numpy as np
from file_params import session_params
from slicingtools import windows_aligned
from window_data import WindowData


class SessionData:
    def __init__(self, filename: str, fps: float = 30):
        self.df = pd.read_csv(filename)
        self.fps = fps

    def split_by_trial(self,
                       trial_type: str = None,
                       pre_period: float = 0,
                       post_period: float = 0,
                       storage_params: str = None) -> WindowData:

        """
        :param trial_type:      Type of trial to be extracted.
                                Acceptable values are specified as the keys of the session_params dict,
                                which is located in file_params.

        :param pre_period:      Time (in seconds) desired to obtain prior to trial onset

        :param post_period:     Time (in seconds) desired to obtain following end of trial

        :param storage_params:  Dict of parameters for saving

        :return:                List of aligned trial data
        """

        # trial_params = session_params[trial_type]
        # print(trial_params)

        if trial_type == 'so':
            indices = np.where(self.df['Shocker on activated'] == 1)[0]

            trials = WindowData(
                                windows_aligned(source_df=self.df,
                                                fps=self.fps,
                                                alignment_points=indices,
                                                pre_frames=int(pre_period * self.fps),
                                                post_frames=int(post_period * self.fps)),
                                fps=self.fps,
                                indices=indices,
                                trial_type=trial_type)

        if storage_params:
            pass

        return trials


if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    my_series = SessionData('C://Users\\limogesaw\\Desktop\\mock_data\\Test_4.csv', fps=8)
    my_trials = my_series.split_by_trial(trial_type='so', pre_period=5, post_period=10)
    print(my_trials.indices)
    #agg_data = np.vstack([window['Speed'].values for window in my_trials.windows])
    #agg_data = pd.DataFrame(agg_data.T, index=my_trials.windows[0].index).T[::-1]
    #ref_range = [np.where(agg_data.T.index.values == 0)[0], np.where(agg_data.T.index.values == 2)[0]]
    agg_data = my_trials.collapse_on(column='Speed')
    agg_data = pd.DataFrame(agg_data, columns=my_trials.windows[0].index)[::-1]
    ref_range = [np.where(agg_data.T.index.values == 0)[0], np.where(agg_data.T.index.values == 2)[0]]
    print(agg_data.shape)
    print(agg_data.T.index.values)
    fig = sns.heatmap(agg_data)
    print(ref_range)
    fig.axvspan(ref_range[0], ref_range[1], color='w', alpha=0.3)
    plt.show()
