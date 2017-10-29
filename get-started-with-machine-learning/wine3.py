from pandas import read_csv
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# load dataset
data = read_csv('winequality-white.csv', header=0, index_col=False, sep=";")

in_data = data.ix[:,:-1].values
in_target = data.ix[:,-1:].values

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(in_data, in_target, test_size=0.2, random_state=50)

#alg = DecisionTreeClassifier()
#alg = KNeighborsClassifier()
#alg = GaussianNB()
#alg = SVC()
alg = RandomForestClassifier()

alg.fit(X_train, Y_train.ravel())
predictions = alg.predict(X_test)
print(accuracy_score(Y_test, predictions))
