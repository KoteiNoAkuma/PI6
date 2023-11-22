#pip install TA-Lib

import talib
import numpy as np
import requests
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


data = None
interval = "1d"
limit = 1000
window_size = 60


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
def calculateRSI(data): # Não mais usado
    return talib.RSI(np.array(data['close']), timeperiod=14)

def calculatePricing(data): 
    return data['close']

def calculateVolume(data): # Não mais usado
    return data['volume']

# Calculo do RSI
rsi_values = calculateRSI(data)

RSI = pd.DataFrame({'RSI': rsi_values}, index=data.index[-len(rsi_values):]) # Dessa forma esse dataframe vai ter o mesmo tamanho dos outros dois dataframes
price = calculatePricing(data)
VOLUME = calculateVolume(data)

history_df = pd.concat([RSI, VOLUME, price], axis=1) # Junta os 3 dataframes em um Único
price = price.dropna().values # Remove as linhas com NaN do dataframe

# Preparando dados para treinamento e teste
X = []
y = []

# Janelamento
for i in range(len(price) - window_size):
    window = price[i:i+window_size] # Elemento 61 nao entra no primeiro loop
    target = price[i+window_size] #  Elemento 60 + i

    X.append(window)
    y.append(target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% treinamento, 20% teste

# Random Forest
def randomForestRegression(X_train, X_test, y_train, y_test):
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_regressor.fit(X, y)
    y_pred = rf_regressor.predict(X)
    preco_previsto = y_pred[-1]

    r2_rf = r2_score(y, y_pred)
    print('=========== Resultados ===========')
    print("Real Data: ")
    print(y)
    print('=========== Previsões ===========')
    print(y_pred)
    print('=========== Preço Previsto ===========')
    print("Preço previsto para o próximo dia:", preco_previsto)
    print('=========== R² ===========')
    print("R² Score:", r2_rf)

    return y_pred, r2_rf



y_pred_rf, r2_rf = randomForestRegression(X_train, X_test, y_train, y_test)

print("Random Forest R2: ", r2_rf) 

# Regressao linear
def linear_regression(X_train, X_test, y_train, y_test):
    regr = LinearRegression()
    regr.fit(X, y)
    y_pred = regr.predict(X)
    preco_previsto = y_pred[-1]

    r2 = r2_score(y, y_pred)
    print('=========== Resultados ===========')
    print("Real Data: ")
    print(y)
    print('=========== Previsões ===========')
    print(y_pred)
    print('=========== Preço Previsto ===========')
    print("Preço previsto para o próximo dia:", preco_previsto)
    print('=========== R² ===========')
    print("R² Score:", r2)


    return y_pred, r2

y_pred, result = linear_regression(X_train, X_test, y_train, y_test)
print("Linear Regression R2: ", result) 
 
# Knn Regression
def knn_regression(X, y, n_neighbors=5):
    knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_regressor.fit(X, y)
    y_pred = knn_regressor.predict(X_test)  # Altere esta linha para usar X_test ao invés de X

    preco_previsto = y_pred[-1]

    r2 = r2_score(y_test, y_pred)  # Use y_test em vez de y

    print('=========== Resultados ===========')
    print("Real Data: ")
    print(y_test)
    print('=========== Previsões ===========')
    print(y_pred)
    print('=========== Preço Previsto ===========')
    print("Preço previsto para o próximo dia:", preco_previsto)
    print('=========== R² ===========')
    print("R² Score:", r2)

    return y_pred, r2

y_pred_knn, r2_knn = knn_regression(X_train, y_train)  # Ajuste a chamada da função com apenas dois argumentos

# SVR Regression
def svr_regression(X, y):
    svr_regressor = SVR(kernel='rbf') 
    svr_regressor.fit(X, y)
    y_pred = svr_regressor.predict(X_test)

    preco_previsto = y_pred[-1]

    r2 = r2_score(y_test, y_pred)

    print('=========== Resultados ===========')
    print("Real Data: ")
    print(y_test)
    print('=========== Previsões ===========')
    print(y_pred)
    print('=========== Preço Previsto ===========')
    print("Preço previsto para o próximo dia:", preco_previsto)
    print('=========== R² ===========')
    print("R² Score:", r2)

    return y_pred, r2

y_pred_svr, r2_svr = svr_regression(X_train, y_train)  
print("SVR R2: ", r2_svr)
