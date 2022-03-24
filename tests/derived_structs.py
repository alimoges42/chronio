import pandas as _pd

if __name__ == '__main__':
    from chronio.structs.raw_structs import BehavioralTimeSeries

    my_series = BehavioralTimeSeries(fpath='C://Users\\limogesaw\\Desktop\\mock_data\\Test_4.csv')
    my_trials = my_series.split_by_trial(trial_type='so', pre_period=5, post_period=30, indices=[100, 400, 700, 1000])

    print(my_trials.data)
    print(str(my_trials))
    print(repr(my_trials))
    print(my_trials.data[0].columns)
    collapsed = my_trials.collapse_on('Speed')

    #collapsed.data = np.random.randint(2, size=(100, 4))
    print(collapsed.data)
    print(str(collapsed))
    print(repr(collapsed))

    #print(collapsed.event_counts())

    print(collapsed.data.index.shape)

    t = _pd.to_timedelta(collapsed.data.index, unit='seconds')
    s = collapsed.data.set_index(t).resample('1S').last().reset_index(drop=True)
    print(t)
    print(s)
    print(collapsed.metadata.session['fps'])