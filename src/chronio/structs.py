from abc import ABC as _ABC
from typing import Any as _Any, List as _List
import numpy as _np
import pandas as _pd
from scipy.interpolate import interp1d as _interp1d
from scipy.stats import sem as _sem

import chronio.analyses as _analyses
from chronio.convention import Convention as _Convention
from chronio.export.exporter import _DataFrameExporter, _ArrayExporter

__all__ = ['Metadata',
           'WindowPane',
           'Window',
           'BehavioralTimeSeries',
           'NeuroTimeSeries']


class Metadata:
    def __init__(self,
                 # System metadata
                 fpath: str = None,

                 # Session metadata
                 stage: str = None,
                 fps: float = None,

                 # Window metadata
                 trial_type: str = None,
                 indices: _List[int] = None,
                 n_windows: int = None,

                 # Dictionary that will overwrite established values
                 meta_dict: dict = None
                 ):
        if not meta_dict:
            meta_dict = {}

        self.system = {'fpath': fpath}
        self.subject = {}
        self.session = {'fps': fps,
                        'stage': stage}
        self.window_params = {'indices': indices,
                              'n_windows': n_windows,
                              'trial_type': trial_type}
        self.pane_params = {}

        self.update(meta_dict=meta_dict)

    def set_val(self, group, value_dict):
        d = getattr(self, group)
        for key, value in value_dict.items():
            d[key] = value

    def update(self, meta_dict: dict):
        for group in list(self.__dict__.keys()):
            self.set_val(group, meta_dict)

    def export(self, convention: _Convention, function_kwargs: dict = None, **exporter_kwargs):
        # TODO: Think about how metadata should be exported
        pass


class _Structure(_ABC):

    def __init__(self, data: _Any, metadata: Metadata):
        self.data = data
        self.metadata = metadata

    def export(self, convention: _Convention, function_kwargs: dict = None, **exporter_kwargs):
        """
        Parameters supplied by exporter_kwargs will replace those supplied by the convention object. This is to
        allow users on-the-fly editing without having to specify all the new fields of a convention object if they
        want to make a minor change to the convention.
        """

        # Overwrites convention params with user-specified export_kwargs (if supplied)
        export_kwargs = convention.get_params()
        export_kwargs.update(exporter_kwargs)

        if type(self.data) == _np.ndarray:
            exporter = _ArrayExporter(obj=self.data, **export_kwargs)

        elif type(self.data) == _pd.DataFrame:
            # TODO: Support both CSV and XLSX export options - could be achieved by editing _DataFrameExporter?
            exporter = _DataFrameExporter(obj=self.data, **export_kwargs)

        else:
            raise ValueError(f'No export protocol for {type(self.data)} exists.')

        if function_kwargs:
            exporter.export(self.data, **function_kwargs)

        else:
            exporter.export(self.data)


class WindowPane(_Structure):
    """
    Class which represents the reduction of a :class: `chronio.structs.derived_structs.Window` on a single feature.
    The data contained in this class correspond to trial-by-trial data of a given feature.

    :param data:        a DataFrame of trial-by-trial data; output by `Window.collapse_on()`
    :type data:         pd.DataFrame

    :param metadata:    metadata for the object
    :type metadata:     :class: `chronio.structs.metadata.Metadata`

    :param pane_params: information about the settings used to obtain the pane.
                        These are then passed to `self.metadata.pane_params`
    :type pane_params:  dict
    """
    def __init__(self, data: _pd.DataFrame, metadata: Metadata, pane_params: dict = None):
        super().__init__(data=data, metadata=metadata)

        if not pane_params:
            pane_params = {}

        for key, value in pane_params:
            setattr(self.metadata.pane_params, key, pane_params[key])

    def __str__(self):
        return f'{self.__class__.__name__} object of shape {self.data.shape}'

    def __repr__(self):
        return f'{self.__class__.__name__}(num_rows={self.data.shape[0]}, num_columns={self.data.shape[1]})'

    def sem(self, axis: int = 0):
        """
        :param axis:    axis along which to compute. By default this is set to 0, which will return the standard error
                        at each timepoint across the window.

        :return:        Returns the standard error of the mean along the specified axis (passed to scipy.stats.sem())
        """

        return _sem(self.data, axis=axis)

    def event_counts(self, axis: int = 0):
        """
        Return counts of events for binarized data. Note that this method requires the variable of interest to
        consist solely of 1s and 0s, where 1s represent event occurrence.

        :param axis:    axis along which to compute. By default this is set to 0, which will count the events in each
                        trial across the window.

        :return:        Returns the total number of events of interest along a given axis in the window
        """

        diff_ = _np.diff(self.data, axis=axis)
        diff_[diff_ == -1] = 0
        return diff_.sum(axis=axis)


class Window(_Structure):
    """
        Class which represents the trial-by-trial data for a given trial of all features
        contained in a time series csv.

        :param data:            a list of DataFrames of trial-by-trial data
        :type data:             List[pd.DataFrame]

        :param metadata:        metadata for the object
        :type metadata:         :class: `chronio.structs.metadata.Metadata`

        :param window_params:   information about the settings used to obtain the pane.
                                These are then passed to `self.metadata.window_params`
        :type window_params:    dict
        """
    def __init__(self,
                 data: _List[_pd.DataFrame],
                 metadata: Metadata,
                 window_params: dict = None):

        super().__init__(data=data, metadata=metadata)

        if not window_params:
            window_params = {}

        for key, value in window_params:
            setattr(self.metadata.window_params, key, window_params[key])

        self.metadata.window_params['n_windows'] = len(self.data)

        if all(window.shape == self.data[0].shape for window in self.data):
            self.metadata.window_params['dims'] = self.data[0].shape

        else:
            raise ValueError("Each DataFrame must have equal shape.")

    def __str__(self):
        return f"{self.__class__.__name__} " \
               f"object containing {self.metadata.window_params['n_windows']} " \
               f"windows of shape {self.metadata.window_params['dims']}"

    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"num_windows={self.metadata.window_params['n_windows']}, " \
               f"dims={self.metadata.window_params['dims']})"

    def collapse_on(self, feature: str) -> WindowPane:
        """
        :param feature:     variable (i.e. column name) from which you would like to extract data

        :return:            WindowPane of the specified feature
        """
        _columns = [f'Trial_{i}' for i in range(1, len(self.data) + 1)]

        agg = _pd.DataFrame(_np.vstack([window[feature].values for window in self.data]).T,
                           columns=_columns,
                           index=self.data[0].index)

        stacked = WindowPane(data=agg,
                             metadata=self.metadata)
        return stacked

    def save_windows(self, save_params: dict):

        pass


class _TimeSeries(_Structure):

    def __init__(self, data: _Any, metadata: Metadata, fpath: str = None,
                 read_csv_kwargs: dict = None):
        super().__init__(data=data, metadata=metadata)

        if fpath:
            self.metadata.system['fpath'] = fpath

            if read_csv_kwargs:
                self.data = _pd.read_csv(self.metadata.system['fpath'], **read_csv_kwargs)

            else:
                self.data = _pd.read_csv(self.metadata.system['fpath'])

        self.metadata.session['fps'] = self.data.shape[0] / round(self.data['Time'].values[-1], 0)

    def downsample_to_length(self, method: str = 'nearest', length: int = None, inplace=False):
        """
        Downsample a dataset to a target length

        :param method:      method of downsampling. Valid methods are 'nearest', 'linear', and 'cut'.
                            For 'nearest' and 'linear' methods, np.interp1d is used, and these methods are passed to
                            the 'kind' parameter of np.interp1d.

        :param length:      desired length of downsampled results

        :param inplace:     if True, updates the self.data parameter to contain only the downsampled dataset.
        """

        x = self.data.index
        y = self.data.values

        f = _interp1d(x, y, axis=0, kind=method)
        x_new = _np.linspace(x[0], x[-1], num=length)
        y_new = f(x_new)
        idx = _np.arange(0, len(y_new), 1)

        if inplace:
            self.data = _pd.DataFrame(index=idx, columns=self.data.columns, data=y_new)

        else:
            return _pd.DataFrame(index=idx, columns=self.data.columns, data=y_new)

    def downsample_by_time(self, interval: float, method: str = 'mean', inplace=False):
        """
        Downsample a dataset by a specified time interval.

        :param interval:    Binning interval (in seconds).

        :param method:      Aggregation method. The 'mean' method will compute the mean value for each interval.

        :param inplace:     if True, updates the self.data parameter to contain only the downsampled dataset.
        """

        # Currently supported methods
        methods = ['mean']

        bins = _np.arange(0, self.data['Time'].values[-1], interval)

        if method == 'mean':
            binned = self.data.groupby(_pd.cut(self.data["Time"], bins)).mean()

        else:
            raise ValueError(f'Method {method} not recognized. Currently accepted methods are {methods}')

        binned.drop(columns=['Time'], inplace=True)
        binned.reset_index(inplace=True)
        binned['Time'] = _np.arange(self.data['Time'].values[0], self.data['Time'].values[-1] - interval, interval)
        binned.set_index('Time')

        if inplace:
            self.data = binned

        else:
            return binned

    def event_onsets(self, cols: list) -> dict:
        return _analyses.event_onsets(self.data, cols=cols)

    def event_intervals(self, cols: list) -> dict:
        return _analyses.event_intervals(self.data, cols=cols)

    def get_streaks(self, cols: list) -> dict:

        """
        :param cols:    Columns on which to compute streaks


        :return:        Dict of dicts, where streaks[col][event] allows access to a list of streak durations.
        """
        streaks = {}

        ieis = _analyses.event_intervals(self.data, cols=cols)

        for col, data in ieis.items():
            streaks[col] = _analyses.streaks_to_lists(streak_df=data)

        return streaks

    def threshold(self, thr: float, binarize: bool = False, columns: _List[str] = None, inplace: bool = False):
        """
        Set a threshold on the specified columns of the dataset and, if desired, binarize the resulting dataset.
        Currently, the same threshold is applied to all specified columns, so if separate thresholds are needed for
        different columns, then this method should be run repeatedly, specifying the target columns for each threshold.

        :param thr:         Threshold value.

        :param binarize:    If True, follow the thresholding with a binarization step that sets all nonzero values to 1.

        :param columns:     List of columns to be thresholded.

        :param inplace:     If True, updates the self.data parameter to contain only the thresholded (or, if applicable,
                            the binarized) dataset.


        :returns:           If inplace == False, returns thresholded (or, if applicable, the binarized) dataset.
                            If inplace == True, modify inplace and return None
        """

        if inplace:
            self.data[columns][self.data[columns] < thr] = 0

            if binarize:
                self.binarize(columns=columns, inplace=inplace)

            return None

        else:
            data = self.data.copy(deep=True)
            data[columns][data[columns] < thr] = 0

            # Can't use self.binarize() here because we made a deep copy
            if binarize:
                data[columns][data[columns] >= thr] = 1

            return data

    def binarize(self, columns: _List[str] = None, inplace: bool = False):
        """
        Binarize a dataset. All nonzero entries (including negative numbers!!) will be set to 1.

        :param columns: Columns to binarize.

        :param inplace: If True, updates the self.data parameter to contain only the binarized dataset.

        :returns:           If inplace == False, returns the binarized dataset.
                            If inplace == True, modify inplace and return None
        """
        if inplace:
            self.data[columns][self.data[columns] != 0] = 1
            return None

        else:
            data = self.data.copy(deep=True)
            data[columns][data[columns] != 0] = 1

            return data

    def split_by_trial(self,
                       indices: list,
                       trial_type: str = None,
                       pre_period: float = 0,
                       post_period: float = 0,
                       storage_params: str = None) -> Window:

        """
        :param indices:         Indices to align to

        :param trial_type:      User-defined name of trial type

        :param pre_period:      Time (in seconds) desired to obtain prior to trial onset

        :param post_period:     Time (in seconds) desired to obtain following end of trial

        :param storage_params:  Dict of parameters for saving

        :return:                List of aligned trial data
        """
        trials = Window(data=_analyses.windows_aligned(source_df=self.data,
                                                       fps=self.metadata.session['fps'],
                                                       alignment_points=indices,
                                                       pre_frames=int(pre_period * self.metadata.session['fps']),
                                                       post_frames=int(post_period * self.metadata.session['fps'])),
                        metadata=self.metadata,
                        window_params={'trial_type': trial_type})

        if storage_params:
            pass

        return trials


class BehavioralTimeSeries(_TimeSeries):

    """
    Time series that represents behavioral data.

    The BehavioralTimeSeries class, unlike the NeuroTimeSeries class, does not assume that all columns
    represent the same form of data, nor that each column is expressed in the same units.
    """
    def __init__(self, data: _Any = None, metadata: Metadata = Metadata(), fpath: str = None):
        super().__init__(data=data, metadata=metadata, fpath=fpath)

    def compute_velocity(self, x_col: str, y_col: str):
        pass


class NeuroTimeSeries(_TimeSeries):

    """
    Time series that represents some form of neural activity data.

    Data in the NeuroTimeSeries class are assumed to be homogeneous. For example, each column may represent a neuron's
    activity, a channel from a recording array, etc.
    """
    def __init__(self, data: _Any = None, metadata: Metadata = Metadata(), fpath: str = None):
        super().__init__(data=data, metadata=metadata, fpath=fpath)

    def correlate(self, axis='cells', method='spearman') -> _Structure:
        methods = ['spearman', 'pearson']
        axes = ['cells', 'time']

        if method not in methods:
            print(f'Correlation method {method} not currently supported.'
                  f'Supported methods are {methods}.')

        if method == 'spearman':
            # TODO: spearman correlation
            pass

        elif method == 'pearson':
            # TODO: pearson correlation
            pass

        return _Structure(data=None, metadata=self.metadata)
