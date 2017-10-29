from pandas import DataFrame

data = DataFrame.from_items(
    [('category', ['Entertainment', 'Lifestyle', 'Technology']),
    ('fb_likes', [2349, 1299, 6589])
    ])

data = data[data['fb_likes'] >= 2000]

print data
