# -*- coding: utf-8 -*-
from pandas import DataFrame
import re

def clean_category(category):
    lower = category.lower()
    x = re.sub(ur'[^a-zA-ZÀ-ú ]+', "", lower, re.UNICODE)
    return x

data = DataFrame.from_items(
    [('category', ['Entertainment:123', 'Lifestyle.456', 'Technology,789']),
    ('fb_likes', [2349, 1299, 6589])
    ])

data['category'] = data['category'].apply(clean_category)

print data
