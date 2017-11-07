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

# sources
# https://github.com/stopwords-iso/stopwords-no
# https://www.dataquest.io/blog/natural-language-processing-with-python/
# https://github.com/luispedro/BuildingMachineLearningSystemsWithPython/blob/master/ch03/rel_post_01.py
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
# Only keep rows with actual engagement, some records had (value|value)
data['Engagement'] = data['Engagement'].apply(prep_engagement)
data['Engagement'] = data['Engagement'].apply(pd.to_numeric, errors='ignore')
data = data[data["Engagement"] > 1000]
data = data[data["Engagement"] < 10000]
# Drops rows without title text and remove special characters etc.
data['Title'] = data['Title'].apply(clean_headline)
data = data[data['Title'].map(len) > 0]

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
targets[targets < targets_mean] = 0
targets[(targets > 0) & (targets > targets_mean)] = 1

selector = SelectKBest(chi2, k=1000)
selector.fit(title_matrix, targets)
top_words = selector.get_support().nonzero()

chi_matrix = title_matrix[:,top_words[0]]

# https://stackoverflow.com/a/14515687
top_ranked_features = sorted(enumerate(selector.scores_),key=lambda x:x[1], reverse=True)[:50]
top_ranked_features_indices = map(list,zip(*top_ranked_features))[0]
for feature_pvalue in zip(np.asarray(vectorizer.get_feature_names())[top_ranked_features_indices],selector.pvalues_[top_ranked_features_indices]):
        print feature_pvalue
