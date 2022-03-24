if __name__ == '__main__':
    import pandas as pd
    from src.chronio import spatial_bins

    data = pd.read_csv('C://Users\\limogesaw\\Desktop\\mock_data\\Test_4.csv')

    fig = plot_positions(data, x_col='Centre position X', y_col='Centre position Y', smoothen=False,
                         downsample_factor=1)
    plt.show()

    H, x_edges, y_edges = spatial_bins(data, x_col='Centre position X', y_col='Centre position Y', bin_size=2,
                                       hist_kwargs={'normed': False})

    fig = spatial_heatmap(H, plot_args={'cmap': 'viridis', 'interpolation': 'bicubic'})
    plt.show()

    data_dict = {}
    for i in range(1, 10):
        data_dict[f'Trial {i}'] = data.iloc[100*i:100*i+100]

    print(data_dict.values())

    # Plot separately
    fig, axs = trial_trajectories(data_dict, x_col='Centre position X', y_col='Centre position Y',
                                  plot_separate=True)
    fig.show()

    # Plot together
    fig, ax = trial_trajectories(data_dict, x_col='Centre position X', y_col='Centre position Y')
    fig.show()