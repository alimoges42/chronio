if __name__ == '__main__':
    from src.chronio import event_intervals, streaks_to_lists
    import numpy as np
    import pandas as pd

    sns.set_theme()
    np.random.seed(5)
    arr = np.random.randint(0, 2, size=(100, 10))
    colnames = [f'Feature {i}' for i in range(0, arr.shape[1])]
    df = pd.DataFrame(arr, columns=colnames)

    print(df.head())
    print(df.columns[9])
    results = event_intervals(source_df=df, cols=[df.columns[9]])
    print(results['Feature 9'])
    to_plot = streaks_to_lists((results['Feature 9']))
    print(to_plot)

    element_map = {0: 'Miss',
                   1: 'Hit'}
    event_histogram(to_plot, element_mapping=element_map, histplot_args={'color': 'red'})
