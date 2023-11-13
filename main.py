#pip install TA-Lib

import talib
import numpy as np
import requests
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

data = None
interval = "1d"
limit = 1000


def get_binance_datarequest(ticker, interval, limit, start='2022-03-01 00:00:00'):
    global data
    columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
    start = int(datetime.datetime.timestamp(pd.to_datetime(start))*1000)
    url = f'https://www.binance.com/api/v3/klines?symbol={ticker}&limit={limit}&interval={interval}&startTime={start}'
    data = pd.DataFrame(requests.get(url).json(), columns=columns, dtype=np.float_)
    data.index = [pd.to_datetime(x, unit='ms').strftime('%Y-%m-%d %H:%M:%S') for x in data.open_time]
    usecols=['open', 'high', 'low', 'close', 'volume', 'qav','num_trades','taker_base_vol','taker_quote_vol']
    data = data[usecols]

get_binance_datarequest('BTCUSDT', interval, limit)

# Calculando RSI usando TA-Lib
def calculateRSI(data):
    return talib.RSI(np.array(data['close']), timeperiod=14)

def calculatePricing(data):
    return data['close']

def calculateVolume(data):
    return data['volume']

# Calculo do RSI
rsi_values = calculateRSI(data)

RSI = pd.DataFrame({'RSI': rsi_values}, index=data.index[-len(rsi_values):]) # Dessa forma esse dataframe vai ter o mesmo tamanho dos outros dois dataframes
PRICING = calculatePricing(data)
VOLUME = calculateVolume(data)

history_df = pd.concat([RSI, VOLUME, PRICING], axis=1) # Junta os 3 dataframes em um Único
history_df = history_df.dropna() # Remove as linhas com NaN do dataframe

# Preparando dados para treinamento e teste
X = history_df[["RSI", "volume"]].values
y = history_df['close'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% treinamento, 20% teste


# Random Forest
def randomForestRegression(X_train, X_test, y_train, y_test):
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)
    r2_rf = r2_score(y_test, y_pred)
    print("Real Data: ")
    print(y_test)
    print('===========')
    print("Predicted Data: ")
    print(y_pred)
    print('===========')

    return y_pred, r2_rf



y_pred_rf, r2_rf = randomForestRegression(X_train, X_test, y_train, y_test)
print("Random Forest R2: ", r2_rf)

# Regressao linear
def linear_regression(X_train, X_test, y_train, y_test):
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("Real Data: ")
    print(y_test)
    print('===========')
    print("Predicted Data: ")
    print(y_pred)
    print('===========')

    return y_pred, r2

y_pred, result = linear_regression(X_train, X_test, y_train, y_test)

print("Linear Regression R2: ", result)

# print(history_df) # Retorna dados de 16 de março de 2022 até 2 de novembro de 2023  597 dias  (Com algumas lacunas)

# # MLP 
# def mlp(X_train, X_test, y_train, y_test):
#     regr = MLPRegressor(random_state=1, max_iter=500)
#     regr.fit(X_train, y_train)
#     y_pred = regr.predict(X_test)
#     r2 = r2_score(y_test, y_pred)

#     print(y_test)
#     print('===========')
#     print(y_pred)

#     print('===========')

#     return y_pred, r2
# y_pred, result = mlp(X_train, X_test, y_train, y_test)

# print(result)




