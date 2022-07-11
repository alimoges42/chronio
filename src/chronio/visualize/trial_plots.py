from typing import Any as _Any, List as _List

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd


__all__ = ['trial_heatmap', 'trial_lineplot']


def trial_heatmap(data: pd.DataFrame,
                  cmap: _Any = 'coolwarm',
                  vline_locs: _List[float] = None,
                  figsize: tuple = (6, 4)):

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize)

    plt.imshow(data.values.T.astype(float),
               extent=[0, data.shape[0], data.shape[1], 0],
               aspect='auto', interpolation='None', cmap=cmap)
    plt.colorbar()

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
