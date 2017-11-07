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

print data
