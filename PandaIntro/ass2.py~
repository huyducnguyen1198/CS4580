import numpy as np
import pandas as pd
df = pd.read_csv('weatherHistory.csv')

a = df.agg({"Temperature (C)":['mean', 'min', 'max']})
print(a)
cond = ["Clear", "Partly Cloudy", "Foggy"]

fil = df['Summary'] == cond[0]
fil = np.logical_or(fil, df['Summary'] == cond[1])
fil = np.logical_or(fil, df['Summary'] == cond[2])

df = df[fil]
gb = df.groupby('Summary')
print(gb.agg({'Temperature (C)': ['mean','min','max']}))

