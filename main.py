#pip install TA-Lib

#import talib
import numpy as np
import requests
import datetime
import pandas as pd
import numpy as np

data = None
interval = "1d"
limit = 1000


def get_binance_datarequest(ticker, interval, limit, start='2021-02-01 00:00:00'):
    global data
    columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
    start = int(datetime.datetime.timestamp(pd.to_datetime(start))*1000)
    url = f'https://www.binance.com/api/v3/klines?symbol={ticker}&limit={limit}&interval={interval}&startTime={start}'
    data = pd.DataFrame(requests.get(url).json(), columns=columns, dtype=np.float_)
    data.index = [pd.to_datetime(x, unit='ms').strftime('%Y-%m-%d %H:%M:%S') for x in data.open_time]
    usecols=['open', 'high', 'low', 'close', 'volume', 'qav','num_trades','taker_base_vol','taker_quote_vol']
    data = data[usecols]

get_binance_datarequest('BTCUSDT', interval, limit)

print(data.to_string())

# Calculando RSI usando TA-Lib
# Arrumar
def calculateRSI(data):
    return talib.RSI(np.array(data['close']), timeperiod=14)

# Media Movel Simples SMA
def calculateSMA(data):
    return talib.SMA(data['close'], timeperiod=20)

def calculatePricing(data):
    return data['close'].mean()

def calculateVolume(data):
    return data['volume'].mean()

# def LSR (API bybit)


