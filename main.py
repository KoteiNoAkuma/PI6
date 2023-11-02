#pip install TA-Lib

import talib
import numpy as np
import requests
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

history_df = pd.concat([RSI, PRICING, VOLUME], axis=1) # Junta os 3 dataframes em um Único
history_df = history_df.dropna() # Remove as linhas com NaN do dataframe

print(history_df) # Retorna dados de 16 de março de 2022 até 2 de novembro de 2023  597 dias 

# Preparando dados para treinamento e test

X = history_df[['RSI', 'volume', 'close']]
y = ...  # Substituir pelo valor do target.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% treinamento, 20% teste

# RNN 
# MLP
# X = rsi, volume, price
# y = target
