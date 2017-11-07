from pandas import read_csv
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
import numpy as np

def good_wine(row):
    if row['quality'] >= 7:
        val = 1
    else:
        val = 0
    return val

# load dataset
data = read_csv('winequality-white.csv', header=0, index_col=False, sep=";")

data['quality'] = data.apply(good_wine, axis=1)

in_data = data.ix[:,:-1].values
in_target = data.ix[:,-1:].values

in_data_prepared = SelectKBest(chi2, k=9).fit_transform(in_data, in_target)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(in_data_prepared, in_target, test_size=0.2, random_state=50)

vals = np.arange(10, 20, 1)
param_grid = dict(n_estimators=vals)

grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, verbose=10)
grid_search.fit(X_train, Y_train.ravel())

print("The best parameters are %s with a score of %0.2f" % (grid_search.best_params_, grid_search.best_score_))
