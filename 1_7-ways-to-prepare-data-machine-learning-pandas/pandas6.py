from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

data = DataFrame.from_items(
    [('category', ['Entertainment', 'Lifestyle', 'Technology']),
    ('fb_likes', [2349, 1299, 6589])
    ])

scaler = MinMaxScaler()
data[['fb_likes']] = scaler.fit_transform(data[['fb_likes']])
print data

#Get unscaled values
data[['fb_likes']] = scaler.inverse_transform(data[['fb_likes']])
