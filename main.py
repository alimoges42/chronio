import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob


f = 'C://Users\\limogesaw\\Desktop\\mock_data\\Test_4.csv'


df = pd.read_csv(f)
print(df.columns)
