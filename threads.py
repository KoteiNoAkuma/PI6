import talib
import numpy as np
import requests
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import concurrent.futures

def get_binance_datarequest(ticker, interval, limit, start='2022-03-01 00:00:00'):
    columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
    start = int(datetime.datetime.timestamp(pd.to_datetime(start))*1000)
    url = f'https://www.binance.com/api/v3/klines?symbol={ticker}&limit={limit}&interval={interval}&startTime={start}'
    data = pd.DataFrame(requests.get(url).json(), columns=columns, dtype=np.float_)
    data.index = [pd.to_datetime(x, unit='ms').strftime('%Y-%m-%d %H:%M:%S') for x in data.open_time]
    usecols=['open', 'high', 'low', 'close', 'volume', 'qav','num_trades','taker_base_vol','taker_quote_vol']
    data = data[usecols]
    return data

def calculateRSI(data): 
    return talib.RSI(np.array(data['close']), timeperiod=14)

def calculatePricing(data): 
    return data['close']

def calculateVolume(data): 
    return data['volume']

def prepare_data(data, window_size=60):
    rsi_values = calculateRSI(data)
    RSI = pd.DataFrame({'RSI': rsi_values}, index=data.index[-len(rsi_values):])
    price = calculatePricing(data)
    VOLUME = calculateVolume(data)
    history_df = pd.concat([RSI, VOLUME, price], axis=1)
    price = price.dropna().values 

    X = []
    y = []

    for i in range(len(price) - window_size):
        window = price[i:i+window_size]
        target = price[i+window_size]

        X.append(window)
        y.append(target)

    return X, y

def predict_next_day(regressor, X, y):
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    predicted_price = y_pred[-1]

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    return y_pred, r2, mae, predicted_price

def predict_next_day_thread(X, y):
    regressors = [
        RandomForestRegressor(n_estimators=100, random_state=0),
        LinearRegression(),
        KNeighborsRegressor(n_neighbors=5),
        SVR(kernel='rbf')
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [executor.submit(predict_next_day, regressor, X, y) for regressor in regressors]

    return [result.result() for result in results]

ticker = 'BTCUSDT'
interval = '1d'
limit = 1000
window_size = 60

data = get_binance_datarequest(ticker, interval, limit)
X, y = prepare_data(data, window_size)

results = predict_next_day_thread(X, y)

for i, (y_pred, r2, mae, predicted_price) in enumerate(results):
    print(f'Model {i + 1}:')
    print('Preço previsto para o próximo dia:', predicted_price)
    print('R² Score:', r2)
    print('Mean Absolute Error (MAE):', mae)
