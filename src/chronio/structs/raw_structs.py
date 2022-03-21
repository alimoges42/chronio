from abc import ABC as _ABC
from typing import Any as _Any, List as _List
import numpy as _np
import pandas as _pd
from scipy.interpolate import interp1d as _interp1d

import chronio.process.analyses as _analyses
from chronio.design.convention import Convention as _Convention
from chronio.structs.metadata import Metadata as _Metadata
from chronio.export.exporter import _DataFrameExporter, _ArrayExporter
from chronio.structs.derived_structs import Window as _Window, _DerivedStructure


class _RawStructure(_ABC):

    def __init__(self, data: _Any, metadata: _Metadata):
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


class _TimeSeries(_RawStructure):

    def __init__(self, data: _Any, metadata: _Metadata, fpath: str = None,
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

        if inplace:
            self.data = _pd.DataFrame(index=x_new, columns=self.data.columns, data=y_new)

        else:
            return _pd.DataFrame(index=x_new, columns=self.data.columns, data=y_new)

    def downsample_by_time(self, interval: float, method: str = 'mean'):
        """
        Downsample a dataset by a specified time interval.

        :param interval:    Binning interval (in seconds).

        :param method:      Aggregation method. The 'mean' method will compute the mean value for each interval.
        """

        # Currently supported methods
        methods = ['mean']

        bins = _np.arange(0, self.data['Time'].values[-1], interval)

        if method == 'mean':
            binned = self.data.groupby(_pd.cut(self.data["Time"], bins)).mean()

        else:
            raise ValueError(f'Method {method} not recognized. Currently accepted methods are {methods}')

        print(binned)
        binned.drop(columns=['Time'], inplace=True)
        binned.reset_index(inplace=True)
        binned['Time'] = _np.arange(self.data['Time'].values[0], self.data['Time'].values[-1] - interval, interval)
        binned.set_index('Time')

        self.data = binned

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
                       storage_params: str = None) -> _Window:

        """
        :param indices:         Indices to align to

        :param trial_type:      User-defined name of trial type

        :param pre_period:      Time (in seconds) desired to obtain prior to trial onset

        :param post_period:     Time (in seconds) desired to obtain following end of trial

        :param storage_params:  Dict of parameters for saving

        :return:                List of aligned trial data
        """
        trials = _Window(data=_analyses.windows_aligned(source_df=self.data,
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

    def __init__(self, data: _Any = None, metadata: _Metadata = _Metadata(), fpath: str = None):
        super().__init__(data=data, metadata=metadata, fpath=fpath)

    def compute_velocity(self, x_col:str, y_col: str):
        pass


class NeuroTimeSeries(_TimeSeries):

    def __init__(self, data: _Any = None, metadata: _Metadata = _Metadata(), fpath: str = None):
        super().__init__(data=data, metadata=metadata, fpath=fpath)

    def correlate(self, axis='cells', method='spearman') -> _DerivedStructure:
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

        return _DerivedStructure(data=None, metadata=self.metadata)


if __name__ == '__main__':
    test = _RawStructure(data=_pd.DataFrame())
    test.export(convention=_Convention(directory='//', suffix='csv', append_date=True, overwrite=False, fields=['name']),
                directory='//aaron/')

    test.export(convention=_Convention(directory='//', suffix='csv', append_date=True, overwrite=False, fields=['name']),
                function_kwargs={'header': False}, fields=['name', 'date'], overwrite=False,
                directory='/home/aaron', suffix='csv')
