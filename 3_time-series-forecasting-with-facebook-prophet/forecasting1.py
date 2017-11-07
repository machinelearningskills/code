import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from fbprophet import Prophet
import datetime
from sklearn.metrics import mean_absolute_error

# source: https://datamarket.com/data/set/2324/daily-minimum-temperatures-in-melbourne-australia-1981-1990
df = pd.read_csv('daily-minimum-temperatures-in-me.csv')

df['ds'] = df['ds'].apply(pd.to_datetime, format='%Y-%m-%d', errors='ignore')

filter_date = datetime.date(1990,01,01)

data_test = df[df['ds'] >= filter_date]
data = df[df['ds'] < filter_date]

m = Prophet()
m.fit(data)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

forecasted = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

components_fig = m.plot_components(forecast)
components_fig.savefig('ts_components.png')

forecast_fig = m.plot(forecast)
forecast_fig.savefig('ts_forecast.png')

#forecasted data for 1990
forecasted_after = forecasted[forecasted['ds'] >= filter_date]

predicted_values = forecasted_after['yhat']
predicted_values_upper = forecasted_after['yhat_upper']
predicted_values_lower = forecasted_after['yhat_lower']
actual_values = data_test['y']

#Calculate the average value of absolute errors as a measurement of accuracy
res = mean_absolute_error(actual_values,predicted_values)
print "Average of the absolute errors: ",res

days = np.arange(0, len(predicted_values), 1)

plt.figure()
plt.figure(figsize=(12,6))
plt.plot(days, actual_values, color='lightskyblue',label="Actual")
plt.plot(days, predicted_values, color='y',label="Predicted")
plt.plot(days, predicted_values_upper, alpha=0.5,linestyle="dotted",color='y',label="Predicted Upper")
plt.plot(days, predicted_values_lower, alpha=0.5,linestyle="dotted",color='y',label="Predicted Lower")
plt.xlabel('Day')
plt.ylabel('Temperature')
plt.legend()
plt.savefig('ts_comparison.png')
plt.show()
