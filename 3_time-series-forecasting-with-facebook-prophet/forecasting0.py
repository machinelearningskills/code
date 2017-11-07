import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# source: https://datamarket.com/data/set/2324/daily-minimum-temperatures-in-melbourne-australia-1981-1990
df = pd.read_csv('daily-minimum-temperatures-in-me.csv')

plt.figure()
plt.figure(figsize=(12,6))
plt.plot(df['y'], color='y')
plt.xlabel('Day')
plt.ylabel('Temperature')
plt.show()
