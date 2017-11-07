from pandas import DataFrame

def is_popuar(row):
    if row['fb_likes'] >= 2000:
        val = 1
    else:
        val = 0
    return val

data = DataFrame.from_items(
    [('category', ['Entertainment', 'Lifestyle', 'Technology']),
    ('fb_likes', [2349, 1299, 6589])
    ])

data['is_popular'] = data.apply(is_popuar, axis=1)

train=data.sample(frac=0.8,random_state=100)
test=data.drop(train.index)

x_train = train.ix[:,:-1]
x_train_target = train.ix[:,-1:]

y_test = test.ix[:,:-1]
y_test_target = test.ix[:,-1:]

print x_train
print x_train_target

#Just use x_train.values and x_train_target.values (same with test data) before fitting
