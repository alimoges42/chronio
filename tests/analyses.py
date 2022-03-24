import numpy as np
import pandas as pd

from chronio.process.analyses import *

if __name__ == '__main__':

    # Simulate dataframe
    np.random.seed(5)
    arr = np.random.randint(0, 2, size=(100, 10))
    colnames = [f'Feature {i}' for i in range(0, arr.shape[1])]
    df = pd.DataFrame(arr, columns=colnames)

    print(df.head())

    # Get event intervals
    intervals = event_intervals(source_df=df, cols=[df.columns[9], df.columns[7]])
    print(intervals['Feature 9'])

    # Use the event intervals as input to get streak data
    print(streaks_to_lists(streak_df=intervals['Feature 9']))

    # Get total durations of each state
    durs = get_state_durations(source_df=df, cols=['Feature 1', 'Feature 3', 'Feature 9'], values=[0], fps=10)
    print(durs)

    # Get onsets of streaks
    ons = event_onsets(source_df=df, cols=['Feature 1', 'Feature 3', 'Feature 9'])
    print(ons)

    # Use indices of event onsets to access timestamps
    df['Time'] = np.linspace(0, 20, df.shape[0])
    print(df)
    print(df.loc[ons['Feature 1'][1]])

    # Testing windows functions
    np.random.seed(5)
    arr = np.random.randint(0, 2, size=(100, 10))
    df = pd.DataFrame(arr)

    print(df.head())

    windows = windows_custom(df, startpoints=[1, 10, 20], endpoints=[4, 15, 40])
    for window in windows:
        print(window)

    windows = windows_aligned(df, alignment_points=[1, 10, 21], pre_frames=5, post_frames=5)
    for window in windows:
        print(window)
    print(windows[0])
    print(len(windows))