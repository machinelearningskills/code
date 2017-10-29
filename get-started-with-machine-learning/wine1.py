from pandas import read_csv
import matplotlib.pyplot as plt

# load dataset
data = read_csv('winequality-white.csv', header=0, index_col=False, sep=";")

data.hist(color="#3F5D7D")
plt.show()

plt.figure(figsize=(9, 6))
plt.scatter(data['pH'].values, data['sulphates'].values, c=data['quality'].values,cmap=plt.cm.cubehelix)
plt.xlabel('Acidiy')
plt.ylabel('Alcohol')
cbar = plt.colorbar()
cbar.set_label("Quality")
plt.show()
