from pandas import DataFrame

data = DataFrame.from_items(
    [('category', ['Entertainment', 'Lifestyle', 'Technology']),
    ('fb_likes', [2349, 1299, 6589])
    ])

train=data.sample(frac=0.8,random_state=100)
test=data.drop(train.index)

print train
print test
