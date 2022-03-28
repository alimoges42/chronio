import pandas as pd
from chronio.structs.structs import BehavioralTimeSeries
from chronio.manage.observations import Session

if __name__ == '__main__':
    f = 'C:\\Users\\limogesaw\\Desktop\\Test_Project\\Test_Experiment\\Test_Cohort1\\Day1\\Mouse1\\Dyn-Cre_CeA_Photometry - Test 42.csv'
    row = pd.Series({'name': 'Aaron', 'behavior file': f, 'ID': '5'})

    obs = Session(row=row, mappings={'behavior file': BehavioralTimeSeries})
    obs.load()

    print(obs.behavior_file)