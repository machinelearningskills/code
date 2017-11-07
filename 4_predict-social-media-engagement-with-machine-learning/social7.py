# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import chi2
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import numpy as np
import nltk.stem
import datetime
import random
import re

# sources
# https://github.com/luispedro/BuildingMachineLearningSystemsWithPython/blob/master/ch03/rel_post_01.py
# https://www.dataquest.io/blog/natural-language-processing-with-python/
# https://github.com/arnauddri/hn

def clean_headline(headline):
    lower = headline.lower()
    x = re.sub(r'[^a-zA-Z\s]', '', lower) #or [^\w\s\d] to save digits as well
    x = re.sub('\s+',' ',x)
    return x

# load dataset
data = read_csv('stories.csv', names=['id', 'created_at', 'created_at_i', 'author', 'points', 'url_hostname', 'num_comments', 'title'])
data = data.drop('id', 1)
data = data.drop('created_at', 1)
data = data.drop('num_comments', 1)
print "Number of rows: ",len(data)
data=data.sample(frac=0.01,random_state=100)

data.dropna(axis=0, how='any')
data['created_at_i'] = data['created_at_i'].apply(pd.to_datetime, unit='s', errors='ignore')
data['points'] = data['points'].apply(pd.to_numeric, errors='ignore')
data = data[data['points'] > 0]
data['title']= data['title'].astype(str)
org_titles = data['title'].copy(deep=True)
data['title'] = data['title'].apply(clean_headline)

print data['title'].head(50)
print "Data shape: ",data.shape

authorEncoder = LabelEncoder()
hostnameEncoder = LabelEncoder()
data['author'] = authorEncoder.fit_transform(data['author'])
data['url_hostname'] = hostnameEncoder.fit_transform(data['url_hostname'])

print "Authors Encoded"

no_stemmer = nltk.stem.SnowballStemmer('english') # see http://www.nltk.org/howto/stem.html for other languages

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([no_stemmer.stem(w) for w in analyzer(doc)])

vectorizer = StemmedCountVectorizer(lowercase=True,stop_words='english')

title_matrix = vectorizer.fit_transform(data['title'])

targets = data[['points']].copy(deep=True)

targets_mean = targets.mean()
#targets_mean = targets_mean*1.5 # you can experiment with this value
targets[targets < targets_mean] = 0
targets[(targets > 0) & (targets > targets_mean)] = 1

selector = SelectKBest(chi2, k=1000)
selector.fit(title_matrix, targets)
top_words = selector.get_support().nonzero()

# https://stackoverflow.com/a/14515687
top_ranked_features = sorted(enumerate(selector.scores_),key=lambda x:x[1], reverse=True)[:25]
top_ranked_features_indices = map(list,zip(*top_ranked_features))[0]
for feature_pvalue in zip(np.asarray(vectorizer.get_feature_names())[top_ranked_features_indices],selector.pvalues_[top_ranked_features_indices]):
        print feature_pvalue

chi_matrix = title_matrix[:,top_words[0]]

title_meta_func = [
    lambda x: len(x),
    lambda x: x.count(" "),
    lambda x: x.count("#"),
    lambda x: x.count("."),
    lambda x: x.count("!"),
    lambda x: x.count("?"),
    lambda x: len(x) / (x.count(" ") + 1),
    lambda x: x.count(" ") / (x.count(".") + 1),
    lambda x: len(re.findall("\d", x)),
    lambda x: len(re.findall("[A-Z]", x)),
]

columns = []
for func in title_meta_func:
    columns.append(org_titles.apply(func))

title_meta = np.asarray(columns).T

posted_meta_func = [
    lambda x: x.year,
    lambda x: x.month,
    lambda x: x.day,
    lambda x: x.hour,
    lambda x: x.minute,
    lambda x: datetime.date(x.year,x.month, x.day).weekday()
]
columns = []
for func in posted_meta_func:
    columns.append(data['created_at_i'].apply(func))

posted_meta = np.asarray(columns).T

author_col = np.asarray(data[['author']])
hostname_col = np.asarray(data[['url_hostname']])

features = np.hstack([posted_meta,title_meta,hostname_col,author_col,chi_matrix.todense()])

random.seed(139710)
indices = list(range(features.shape[0]))
random.shuffle(indices)
train_rows = int(len(data)*0.8)

train = features[indices[:train_rows], :]
test = features[indices[train_rows:], :]
train_target = data['points'].iloc[indices[:train_rows]]
test_target = data['points'].iloc[indices[train_rows:]]
train = np.nan_to_num(train)

print "Starting training"

reg = Ridge(alpha=.4)
#reg = KernelRidge()

reg.fit(train, train_target)
predictions = reg.predict(test)
res = reg.score(test, test_target)
print "Accuracy: ",res

average_predictions = sum(predictions)/len(predictions)
mse = sum(abs(predictions - test_target)) / len(predictions)
print "Average diff in upvotes (predicted vs actual):", mse

average_upvotes = sum(test_target)/len(test_target)
avg = sum(abs(average_upvotes - test_target)) / len(predictions)
print "Average in dataset: ",avg
# vals = np.arange(0.1, 10, 0.1)
# param_grid = dict(alpha=vals)
#
# grid_search = GridSearchCV(Ridge(), param_grid=param_grid, verbose=10)
# grid_search.fit(train, train_target)
#
# print("The best parameters are %s with a score of %0.2f" % (grid_search.best_params_, grid_search.best_score_))

vals = np.arange(0, len(test_target), 1)
samples = 100 #adjust according to your data.

plt.figure()
p1 = plt.bar(vals[:samples], test_target[:samples], 0.5, color='lightskyblue',label="Upvotes")
p2 = plt.bar(vals[:samples], predictions[:samples], 0.5, alpha=0.5,color='y',label="Predictions")
plt.plot([0, samples], [average_upvotes, average_upvotes], color='lightskyblue', linestyle='-', linewidth=1, label="Average Upvotes")
plt.plot([0, samples], [average_predictions, average_predictions], color='y', linestyle='-', linewidth=1, label="Average Predictions")
plt.xlabel('Test')
plt.ylabel('Upvotes')
plt.legend()
plt.show()
