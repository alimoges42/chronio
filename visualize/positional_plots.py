import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def plot_positions(x_coords: np.array,
                   y_coords: np.array,
                   smoothen: bool = False,
                   downsample_factor: int = 1,
                   plot_args: dict = None):

    if len(x_coords) != len(y_coords):
        raise ValueError('Lengths of coordinates must match')

    if plot_args is None:
        plot_args = {}

    x_coords = x_coords[0::downsample_factor]
    y_coords = y_coords[0::downsample_factor]

    if smoothen:
        from scipy.interpolate import splprep, splev

        def interpolate_polyline(polyline, num_points):
            duplicates = []
            for i in range(1, len(polyline)):
                if np.allclose(polyline[i], polyline[i - 1]):
                    duplicates.append(i)
            if duplicates:
                polyline = np.delete(polyline, duplicates, axis=0)
            tck, u = splprep(polyline.T, s=0)
            u = np.linspace(0.0, 1.0, num_points)
            return np.column_stack(splev(u, tck))
        x_coords = x_coords.values[~np.isnan(x_coords.values)]
        y_coords = y_coords.values[~np.isnan(y_coords.values)]

        dat = np.concatenate([x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)], axis=1)

        results = interpolate_polyline(dat, len(dat))
        x_coords = results[:, 0]
        y_coords = results[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.plot(x_coords, y_coords)
    return fig


def spatial_heatmap(spatial_data, q: float = 0.95, blur_sigma: float = 1, plot_args: dict = None):

    vmax = np.quantile(spatial_data[np.nonzero(spatial_data)], q=q)

    spatial_data = gaussian_filter(spatial_data, sigma=blur_sigma)

    if plot_args is None:
        fig = plt.imshow(spatial_data, vmax=vmax, cmap='inferno', interpolation='bicubic')

    else:
        fig = plt.imshow(spatial_data, vmax=vmax, **plot_args)
    return fig


def trial_trajectories(data: dict):
    pass


if __name__ == '__main__':
    import pandas as pd
    from analyses import spatial_bins

    data = pd.read_csv('C://Users\\limogesaw\\Desktop\\mock_data\\Test_4.csv')
    my_coords = data[['Centre position X', 'Centre position Y']]

    fig = plot_positions(my_coords['Centre position X'], my_coords['Centre position Y'], smoothen=False,
                         downsample_factor=1)
    plt.show()

    H, x_edges, y_edges = spatial_bins(data, x_col='Centre position X', y_col='Centre position Y', bin_size=2,
                                       hist_kwargs={'normed': False})

    fig = spatial_heatmap(H, plot_args={'cmap': 'viridis', 'interpolation': 'bicubic'})
    plt.show()

    print(data)
