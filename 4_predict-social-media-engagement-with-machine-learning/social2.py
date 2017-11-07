# -*- coding: utf-8 -*-
from pandas import read_csv
import pandas as pd
import re
import numpy

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
print "Original shape: ",data.shape
print data.dtypes
# we don't need ID at this time
data = data.drop('ID', 1)
# Some dates had negative value, remove these rows (probably drafts)
data = data[data['Posted'] > 0]
# Only keep rows with actual engagement, some records had (value|value)
data['Engagement'] = data['Engagement'].apply(prep_engagement)
data['Engagement'] = data['Engagement'].apply(pd.to_numeric, errors='ignore')
data = data[data["Engagement"] > 0]
# Drops rows without title text and remove special characters etc.
data['Title'] = data['Title'].apply(clean_headline)
data = data[data['Title'].map(len) > 0]

print "Shape after: ",data.shape
print data.dtypes
