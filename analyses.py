import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import groupby
from typing import List, Any


def event_onsets(source_df: pd.DataFrame,
                 cols: list) -> dict:
    """
    Obtains the frame numbers that signify onsets of events in desired columns of events.

    :param source_df:   original dataframe to use

    :param cols:        list of columns of the dataframe for which you would like IEIs to be computed

    :return:            dict of dicts containing the frame numbers corresponding to each onset of an event
                        for each specified column
    """
    onsets = {}
    for col in cols:

        onsets[col] = {}

        col_states = source_df[col].to_frame()
        col_states['streak_start'] = col_states[col].ne(col_states[col].shift()).astype(str)
        col_states = col_states.groupby(col)

        for state, group in col_states:
            onsets[col][state] = group[group['streak_start'] == 'True'].index.values

    return onsets


def event_intervals(source_df: pd.DataFrame,
                    cols: list) -> dict:
    """
    Compute the inter-event intervals (IEIs) for desired columns of a DataFrame.

    :param source_df:   original dataframe to use

    :param cols:        list of columns of the dataframe for which you would like IEIs to be computed

    :return:            dict of event interval data for each specified column
    """

    results = defaultdict(pd.DataFrame)

    for col in cols:
        data = {'Elements': [], 'Counts': []}

        streaks = [list(group) for key, group in groupby(source_df[col].values.tolist())]

        for streak in streaks:
            data['Elements'].append(streak[0])
            data['Counts'].append(len(streak))

        results[col] = pd.DataFrame.from_dict(data=data)

    return results


def streaks_to_lists(streak_df: pd.DataFrame):
    """
    :param streak_df:   A DataFrame of streaks with the column names of 'Elements' and 'Streaks'.
                        These correspond to the DataFrames contained in the list returned by
                        the event_intervals function.

    :return:            Returns elements_dict, a dictionary of each unique element found within the
                        'Elements' column of the streak_df. Each value is simply the value originally
                        contained within the 'Counts' column.
    """

    elements_dict = {}
    for uniq in streak_df['Elements'].unique():
        elements_dict[uniq] = streak_df.loc[streak_df['Elements'] == uniq]['Counts'].tolist()

    return elements_dict


def spatial_bins(source_df: pd.DataFrame,
                 x_col: str,
                 y_col: str,
                 bin_size: float = 1,
                 handle_nans: str = 'bfill',
                 hist_kwargs: dict = None
                 ) -> tuple:
    """
    :param source_df: df containing columns for x and y coordinates
    :param x_col: name of the column with x data
    :param y_col: name of the column with y data
    :param bin_size: bin size (in mm)
    :param handle_nans: valid options are 'bfill', 'ffill', or 'drop'
    :param hist_kwargs: optional kwargs for np.histogram2d
    :return:
    """
    if hist_kwargs is None:
        hist_kwargs = {}

    if handle_nans == 'drop':
        source_df = source_df.dropna()

    else:
        source_df = source_df.fillna(method=handle_nans)

    # Compute number of x and y bins needed at the current bin_size
    _x_bins = (source_df[x_col].max() - source_df[x_col].min()) / bin_size
    x_bins = int((_x_bins + bin_size) - (_x_bins % bin_size))

    _y_bins = (source_df[y_col].max() - source_df[y_col].min()) / bin_size
    y_bins = int((_y_bins + bin_size) - (_y_bins % bin_size))

    # Extract 2D histogram
    H, x_edges, y_edges = np.histogram2d(source_df[x_col].values,
                                         source_df[y_col].values,
                                         bins=[x_bins, y_bins],
                                         **hist_kwargs)
    H = H.T

    return H, x_edges, y_edges


def get_state_durations(source_df: pd.DataFrame,
                        cols: List[str],
                        values: list = None,
                        fps: float = 30) -> pd.DataFrame:
    """
    Computes the durations of states in specific columns of a source DataFrame.
    Unique values of supplied dataframes are interpreted as unique


    :param source_df:   DataFrame

    :param cols:        Columns for which the durations are to be computed

    :param values:      Values for which duration will be computed. If None, will extract duration for all states

    :param fps:         Imaging rate (frames per second) of df

    :return:            DataFrame whose columns are equivalent to "cols" parameter and whose rows are either
                            1) a list of corresponding to "values" parameter (if provided), or
                            2) all unique states found across all columns.

                        Each element is the duration of each state (in seconds).

    """

    durations = pd.DataFrame()

    for col in cols:
        col_durations = source_df[col].value_counts() / fps

        if values:
            durations[col] = col_durations[values]

        else:
            durations[col] = col_durations

    return durations


if __name__ == '__main__':
    np.random.seed(5)
    arr = np.random.randint(0, 2, size=(100, 10))
    colnames = [f'Feature {i}' for i in range(0, arr.shape[1])]
    df = pd.DataFrame(arr, columns=colnames)

    print(df.head())
    intervals = event_intervals(source_df=df, cols=[df.columns[9]])
    print(intervals['Feature 9'])
    print(streaks_to_lists(streak_df=intervals['Feature 9']))

    durs = get_state_durations(source_df=df, cols=['Feature 1', 'Feature 3', 'Feature 9'], values=[0], fps=10)
    print(durs)

    ons = event_onsets(source_df=df, cols=['Feature 1', 'Feature 3', 'Feature 9'])
    print(ons)

