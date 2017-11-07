from pandas import read_csv
from sklearn import model_selection

# load dataset
data = read_csv('winequality-white.csv', header=0, index_col=False, sep=";")


in_data = data.ix[:,:-1].values
in_target = data.ix[:,-1:].values

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(in_data, in_target, test_size=0.2, random_state=50)
