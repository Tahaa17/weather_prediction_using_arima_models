import datetime, warnings, scipy 
import pandas as pd
import numpy as np
import math
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
dataset = pd.read_csv('./data/weather_data_toronto.csv', index_col="Date/Time")

dataset.drop(["Longitude (x)","Latitude (y)","Station Name","Climate ID","Year","Month","Day","Data Quality","Max Temp Flag","Min Temp Flag","Mean Temp Flag","Heat Deg Days (°C)","Heat Deg Days Flag","Cool Deg Days (°C)","Cool Deg Days Flag","Total Rain (mm)","Total Rain Flag","Total Snow (cm)","Total Snow Flag","Total Precip (mm)","Total Precip Flag","Snow on Grnd (cm)","Snow on Grnd Flag","Dir of Max Gust (10s deg)","Dir of Max Gust Flag","Spd of Max Gust (km/h)","Spd of Max Gust Flag", "Min Temp (°C)", "Max Temp (°C)"], axis=1,inplace=True)
dataset=dataset.dropna()
print(dataset)

dataset['Mean Temp (°C)'].plot(figsize=(12,5), ylabel='Mean Temp (°C)')
pyplot.show()
split_point = int(724)
train, validation = dataset[0:split_point], dataset[split_point:]
print('Training Data %d, Validation %d' % (len(train), len(validation)))
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

X = train.values
days_in_year = 365
differenced = difference(X, days_in_year)
model = ARIMA(differenced, order=(1,1,2))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())

forecast = model_fit.forecast(steps=727)
history = [x for x in X]
starting_date = datetime.datetime.strptime("21-01-01", "%y-%m-%d").date()
print(starting_date)
day = 1
df = pd.DataFrame(columns=['Date/Time','Mean Temp (°C)'])

for yhat in forecast:
    inverted = inverse_difference(history, yhat, days_in_year)
    #print('Day %d: %f' % (day, inverted))
    prediction = {'Date/Time':starting_date, 'Mean Temp (°C)':inverted[0]}
    df = df.append(prediction, ignore_index=True)
    history.append(inverted)
    day += 1
    starting_date=starting_date+datetime.timedelta(days=1)

print(df.head())
df.to_csv('predictions.csv', index=False)
print("VALIDATION")
print(validation.head())

validation['Mean Temp (°C)'].plot(figsize=(20,10))
model_prediction = df.iloc[0:362]
print(model_prediction)
model_prediction['Mean Temp (°C)'].plot(figsize=(20,10),color='orange',ylabel='Mean Temp (°C)')
pyplot.show()
#df['Mean Temp (°C)'].plot(figsize=(20,10),color='orange')
validation['Mean Temp (°C)'].plot(figsize=(20,10))
df['Mean Temp (°C)'].plot(figsize=(20,10),color='orange', ylabel='Mean Temp (°C)')
pyplot.show()
df=df.set_index('Date/Time')
future=df.iloc[362:]
future['Mean Temp (°C)'].plot(figsize=(20,10),color='orange', ylabel='Mean Temp (°C)')
pyplot.show()