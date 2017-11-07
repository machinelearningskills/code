# -*- coding: utf-8 -*-
from pandas import read_csv

# load dataset
data = read_csv('data-export.csv', header=0, index_col=False, sep=';', encoding='utf-8')
print "Original shape: ",data.shape
print data.dtypes
