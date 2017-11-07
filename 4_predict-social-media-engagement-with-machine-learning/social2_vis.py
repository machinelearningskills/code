# -*- coding: utf-8 -*-
from pandas import read_csv
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

def prep_engagement(engagement):
    x = str(engagement).split('|')
    if x[0]=='nan':
        return 0;
    else:
        return x[0];

# load dataset
data = read_csv('data-export.csv', header=0, index_col=False, sep=';', encoding='utf-8')
data = data[data['Posted'] > 0]
data['Engagement'] = data['Engagement'].apply(prep_engagement)
data['Engagement'] = data['Engagement'].apply(pd.to_numeric, errors='ignore')

#drop rows with more than 10.000 engagement
data = data[data["Engagement"] < 10000]

vals = np.arange(0, len(data), 1)

plt.figure(figsize=(9, 6))
plt.scatter(vals,data['Engagement'], s=1)
plt.xlabel('Rows')
plt.ylabel('Engagement')
plt.show()
