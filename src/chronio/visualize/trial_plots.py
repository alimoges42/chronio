from typing import Any as _Any, List as _List

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, CenteredNorm
import matplotlib.cm as cm
from seaborn import set_context
import pandas as pd
import numpy as np


__all__ = ['trial_heatmap', 'trial_lineplot']


def _keep_center_colormap(cmap, vmin, vmax, center=0):
    vmin = vmin - center
    vmax = vmax - center
    dv = max(-vmin, vmax) * 2
    N = int(256 * dv / (vmax-vmin))
    cmap = cm.get_cmap(cmap, N)
    newcolors = cmap(np.linspace(0, 1, N))
    beg = int((dv / 2 + vmin)*N / dv)
    end = N - int((dv / 2 - vmax)*N / dv)
    newmap = ListedColormap(newcolors[beg:end])
    return newmap


def trial_heatmap(data: _Any,
                  fps: float = None,
                  cmap: _Any = 'seismic',
                  center_cmap: bool = True,
                  axis_mode: str = 'time',
                  vline_locs: _List[float] = None,
                  figsize: tuple = (6, 4),
                  context: str = None):

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    if context:
        set_context(context)

    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(data, pd.DataFrame):
        to_plot = data.values

        if axis_mode == 'time':
            extent = [data.index[0], data.index[-1], data.shape[1], 0]
            xlabel = 'Time (s)'
            print(extent)

    elif isinstance(data, np.ndarray):
        to_plot = data
        if axis_mode == 'time':
            extent = [0, round((data.shape[0]) / fps, 3), data.shape[1], 0]
            xlabel = 'Time (s)'
            print(extent)

    else:
        raise TypeError(f'Invalid input of type {type(data)}. Only np.ndarray and pd.DataFrame objects accepted.')

    if center_cmap is True:
        cmap = _keep_center_colormap(cmap=cmap, vmin=np.min(to_plot), vmax=np.max(to_plot), center=0)

    if axis_mode == 'frame':
        extent = [0, data.shape[0], data.shape[1], 0]
        xlabel = 'Frame number'

    plt.imshow(to_plot.T.astype(float),
               extent=extent,
               aspect='auto', interpolation='None', cmap=cmap)

    plt.colorbar()
    plt.xlabel(xlabel)

    if vline_locs:
        for vline_loc in vline_locs:
            ax.axvline(vline_loc, c='k', ls='--')
    ax.axes.get_yaxis().set_visible(True)

    return fig, ax


def trial_lineplot(data: pd.DataFrame,
                   cmap: _Any = 'bone',
                   vline_loc: float = None,
                   figsize: tuple = (6, 4)):

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap, len(data))

    if type(cmap) == LinearSegmentedColormap:
        colors = cmap(np.arange(0, cmap.N))

    elif type(cmap) == ListedColormap:
        colors = cmap.colors

    else:
        colors = cmap

    fig, ax = plt.subplots(figsize=figsize)

    for row, vals in enumerate(data.values):
        ax.plot(data.columns, vals, color=colors[row], lw=3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(data)))
    cbar = plt.colorbar(sm)
    ax.axvline(vline_loc, c='k', ls='--')

    return fig, ax, cbar


if __name__ == '__main__':
    import numpy as np

    np.random.seed(0)
    data = np.random.randn(20, 30)

    data = pd.DataFrame(data=data, index=np.arange(0, len(data)),
                        columns=np.arange(0, len(data.T)))

    fig, ax = trial_heatmap(data=data, vline_locs=[8])

    cbar = None
    #fig, ax, cbar = trial_lineplot(data=data, vline_locs=[8])
    ax.tick_params(labelsize=14)
    if cbar:
        cbar.ax.tick_params(labelsize=14)
    fig.show()
