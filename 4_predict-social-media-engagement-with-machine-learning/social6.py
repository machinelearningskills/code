# -*- coding: utf-8 -*-
from pandas import read_csv
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
import nltk.stem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import datetime
import random
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import BayesianRidge

# sources
# https://github.com/stopwords-iso/stopwords-no
# https://github.com/luispedro/BuildingMachineLearningSystemsWithPython/blob/master/ch03/rel_post_01.py
# https://www.dataquest.io/blog/natural-language-processing-with-python/
stopwords = ['alle','andre','arbeid','at','av','bare','begge','ble','blei','bli','blir','blitt','bort','bra','bruke','både','båe','da','de','deg','dei','deim','deira','deires','dem','den','denne','der','dere','deres','det','dette','di','din','disse','ditt','du','dykk','dykkar','då','eg','ein','eit','eitt','eller','elles','en','ene','eneste','enhver','enn','er','et','ett','etter','folk','for','fordi','forsûke','fra','få','før','fûr','fûrst','gjorde','gjûre','god','gå','ha','hadde','han','hans','har','hennar','henne','hennes','her','hjå','ho','hoe','honom','hoss','hossen','hun','hva','hvem','hver','hvilke','hvilken','hvis','hvor','hvordan','hvorfor','i','ikke','ikkje','ingen','ingi','inkje','inn','innen','inni','ja','jeg','kan','kom','korleis','korso','kun','kunne','kva','kvar','kvarhelst','kven','kvi','kvifor','lage','lang','lik','like','makt','man','mange','me','med','medan','meg','meget','mellom','men','mens','mer','mest','mi','min','mine','mitt','mot','mye','mykje','må','måte','navn','ned','nei','no','noe','noen','noka','noko','nokon','nokor','nokre','ny','nå','når','og','også','om','opp','oss','over','part','punkt','på','rett','riktig','samme','sant','seg','selv','si','sia','sidan','siden','sin','sine','sist','sitt','sjøl','skal','skulle','slik','slutt','so','som','somme','somt','start','stille','så','sånn','tid','til','tilbake','tilstand','um','under','upp','ut','uten','var','vart','varte','ved','verdi','vere','verte','vi','vil','ville','vite','vore','vors','vort','vår','være','vært','vöre','vört','å']

def clean_headline(headline):
    lower = headline.lower()
    x = re.sub(ur'[^a-zA-ZÀ-ú ]+', "", lower, re.UNICODE)
    x = re.sub('\\s+',' ',x, re.UNICODE)
    return x

def prep_engagement(engagement):
    x = str(engagement).split('|')
    if x[0]=='nan':
        return 0;
    else:
        return x[0];

# load dataset
data = read_csv('data-export.csv', header=0, index_col=False, sep=';', encoding='utf-8')
# we don't need ID at this time
data = data.drop('ID', 1)
# Some dates had negative value, remove these rows (probably drafts)
data = data[data['Posted'] > 0]
data['Posted'] = data['Posted'].apply(pd.to_datetime, format='%Y%m%d', errors='ignore')

# filter data based on date
filter_date = datetime.date(2016,01,01)
data = data[data['Posted'] > filter_date]

# Only keep rows with actual engagement, some records had (value|value)
data['Engagement'] = data['Engagement'].apply(prep_engagement)
data['Engagement'] = data['Engagement'].apply(pd.to_numeric, errors='ignore')
data = data[data["Engagement"] > 0]
data = data[data["Engagement"] < 10000]
# Drops rows without title text and remove special characters etc.
data = data[data['Title'].map(len) > 5]
org_titles = data['Title'].copy(deep=True)
data['Title'] = data['Title'].apply(clean_headline)

print "Data shape: ",data.shape

typeEncoder = LabelEncoder()
data['Type'] = typeEncoder.fit_transform(data['Type'])

no_stemmer = nltk.stem.SnowballStemmer('norwegian') # see http://www.nltk.org/howto/stem.html for other languages

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([no_stemmer.stem(w) for w in analyzer(doc)])

vectorizer = StemmedCountVectorizer(lowercase=True,stop_words=stopwords)

title_matrix = vectorizer.fit_transform(data['Title'])

targets = data[['Engagement']].copy(deep=True)

targets_mean = targets.mean()
targets_mean = targets_mean*1.5
targets[targets < targets_mean] = 0
targets[(targets > 0) & (targets > targets_mean)] = 1

selector = SelectKBest(chi2, k=400)
selector.fit(title_matrix, targets)
top_words = selector.get_support().nonzero()

chi_matrix = title_matrix[:,top_words[0]]

title_meta_func = [
    lambda x: len(x),
    lambda x: x.count(" "),
    lambda x: x.count("."),
    lambda x: x.count("!"),
    lambda x: x.count("?"),
    lambda x: len(x) / (x.count(" ") + 1),
    lambda x: x.count(" ") / (x.count(".") + 1),
    lambda x: len(re.findall(u"\d", x, re.UNICODE)),
    lambda x: len(re.findall(u"[a-zA-ZÀ-ú]", x, re.UNICODE)),
]

columns = []
for func in title_meta_func:
    columns.append(org_titles.apply(func))

title_meta = np.asarray(columns).T

posted_meta_func = [
    #lambda x: x.year,
    #lambda x: x.month,
    #lambda x: x.day,
    #lambda x: x.hour,
    #lambda x: x.minute,
    lambda x: datetime.date(x.year,x.month, x.day).weekday()
]
columns = []
for func in posted_meta_func:
    columns.append(data['Posted'].apply(func))

posted_meta = np.asarray(columns).T

type_col = np.asarray(data[['Type']])

features = np.hstack([posted_meta,title_meta,type_col,chi_matrix.todense()])

random.seed(10)
indices = list(range(features.shape[0]))
random.shuffle(indices)
train_rows = int(len(data)*0.8)

train = features[indices[:train_rows], :]
test = features[indices[train_rows:], :]
train_target = data['Engagement'].iloc[indices[:train_rows]]
test_target = data['Engagement'].iloc[indices[train_rows:]]
train = np.nan_to_num(train)

#reg = Ridge()
#reg = KernelRidge()
reg = BayesianRidge()

reg.fit(train, train_target)
predictions = reg.predict(test)
res = reg.score(test, test_target)
print "Accuracy: ",res

mse = sum(abs(predictions - test_target)) / len(predictions)
print "Average diff in engagement (predicted vs actual):", mse

# vals = np.arange(300, 400, 10)
# param_grid = dict(n_iter=vals)
#
# grid_search = GridSearchCV(BayesianRidge(), param_grid=param_grid, verbose=10)
# grid_search.fit(train, train_target)
#
# print("The best parameters are %s with a score of %0.2f" % (grid_search.best_params_, grid_search.best_score_))
import matplotlib.pyplot as plt

vals = np.arange(0, len(test_target), 1)
samples = 100 #adjust according to your data.

plt.figure()
p1 = plt.bar(vals[:samples], test_target[:samples], 0.5, color='lightskyblue',label="Engagement")
p2 = plt.bar(vals[:samples], predictions[:samples], 0.5, alpha=0.5,color='y',label="Predictions")
plt.xlabel('Test')
plt.ylabel('Engagement')
plt.legend()
plt.show()
