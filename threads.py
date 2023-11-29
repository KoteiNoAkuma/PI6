import talib
import numpy as np
import requests
import datetime
import pandas as pd
import numpy as np
import threading
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from tkinter import *
import tkinter.font as tkFont
from sklearn.neural_network import MLPRegressor


def get_binance_datarequest(ticker, interval, limit, start='2022-03-01 00:00:00'):
    columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
    start = int(datetime.datetime.timestamp(pd.to_datetime(start))*1000)
    url = f'https://www.binance.com/api/v3/klines?symbol={ticker}&limit={limit}&interval={interval}&startTime={start}'
    data = pd.DataFrame(requests.get(url).json(), columns=columns, dtype=np.float_)
    data.index = [pd.to_datetime(x, unit='ms').strftime('%Y-%m-%d %H:%M:%S') for x in data.open_time]
    usecols=['open', 'high', 'low', 'close', 'volume', 'qav','num_trades','taker_base_vol','taker_quote_vol']
    data = data[usecols]

    return data

def calculatePricing(data): 
    return data['close']


def prepare_data(data, window_size=60):
    price = calculatePricing(data)
    price = price.dropna().values
    
    previous_close = data['close'].iloc[-1]  # Valor de fechamento do dia atual


    X = []
    y = []

    for i in range(len(price) - window_size):
        window = price[i:i+window_size]
        target = price[i+window_size]

        X.append(window)
        y.append(target)

    simulated_next_day = np.append(price[-window_size+1:], previous_close)  # Simula o próximo dia usando o último preço conhecido como fechamento


    return X, y, previous_close, simulated_next_day

def randomForestRegression(X_train, y_train, last_known_data):
    rf_regressor = RandomForestRegressor(n_estimators=200, random_state=0)
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_train)
    
    next_day_point = last_known_data.reshape(1, -1)  
    predicted_price = rf_regressor.predict(next_day_point)[0]

    r2 = r2_score(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)

    thread = threading.Thread(target=print_results, args=('Random Forest', predicted_price, r2, mae))
    thread.start()

    return y_pred, r2, mae, predicted_price

def linear_regression(X_train, y_train, last_known_data):
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_train)
    
    predicted_price = regr.predict(last_known_data.reshape(1, -1))[0]

    r2 = r2_score(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)

    thread = threading.Thread(target=print_results, args=('Linear Regression', predicted_price, r2, mae))
    thread.start()

    return y_pred, r2, mae, predicted_price

def knn_regression(X_train, y_train, last_known_data, n_neighbors=5):
    knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_regressor.fit(X_train, y_train)
    y_pred = knn_regressor.predict(X_train)
    
    predicted_price = knn_regressor.predict(last_known_data.reshape(1, -1))[0]

    r2 = r2_score(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)

    thread = threading.Thread(target=print_results, args=('KNN Regression', predicted_price, r2, mae))
    thread.start()

    return y_pred, r2, mae, predicted_price

def svr_regression(X_train, y_train, last_known_data):
    svr_regressor = SVR(kernel='rbf')
    svr_regressor.fit(X_train, y_train)
    y_pred = svr_regressor.predict(X_train)
    
    predicted_price = svr_regressor.predict(last_known_data.reshape(1, -1))[0]

    r2 = r2_score(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)

    thread = threading.Thread(target=print_results, args=('SVR Regression', predicted_price, r2, mae))
    thread.start()

    return y_pred, r2, mae, predicted_price

def print_results(algorithm_name, predicted_price, r2, mae):
    print(f'{algorithm_name}:')
    print('Preço previsto para o próximo dia:', predicted_price)
    print('R² Score:', r2)
    print('Mean Absolute Error (MAE):', mae)


def design(predicted_rf, r2_rf, mae_rf, predicted_lr, r2_lr, mae_lr, predicted_knn, r2_knn, mae_knn, predicted_svr, r2_svr, mae_svr, previous_close):
    top = Tk()
    top.geometry("700x450")
    fontExample = tkFont.Font(family="Arial", size=9, weight="bold", slant="italic")
    algoritmo = Label(top, text="ALGORITMO", font=fontExample).place(x=30, y=50)
    erro = Label(top, text="MAE", font=fontExample).place(x=180, y=50)
    valorPrevisto = Label(top, text="VALOR PREVISTO", font=fontExample).place(x=280, y=50)
    r2 = Label(top, text="R2", font=fontExample).place(x=465, y=50)
    valorAtual = Label(top, text="VALOR ATUAL", font=fontExample).place(x=580, y=50)
    randomForest = Label(top, text="Random\nForest").place(x=30, y=120)
    linearRegression = Label(top, text="Linear\nRegression").place(x=30, y=180)
    knnRegression = Label(top, text="KNN\nRegression").place(x=30, y=240)
    svrRegression = Label(top, text="SVR\nRegression").place(x=30, y=300)
    error1 = Label(top, text=str(round(mae_rf, 2))).place(x=180, y=120)
    error2 = Label(top, text=str(round(mae_lr, 2))).place(x=180, y=180)
    error3 = Label(top, text=str(round(mae_knn, 2))).place(x=180, y=240)
    error4 = Label(top, text=str(round(mae_svr, 2))).place(x=180, y=300)
    value_predict1 = Label(top, text=str(round(predicted_rf, 2))).place(x=310, y=120)
    value_predict2 = Label(top, text=str(round(predicted_lr, 2))).place(x=310, y=180)
    value_predict3 = Label(top, text=str(round(predicted_knn, 2))).place(x=310, y=240)
    value_predict4 = Label(top, text=str(round(predicted_svr, 2))).place(x=310, y=300)
    r2_Rf = Label(top, text=str(round(r2_rf, 2))).place(x=450, y=120)
    r2_Lr = Label(top, text=str(round(r2_lr, 2))).place(x=450, y=180)
    r2_Knn = Label(top, text=str(round(r2_knn, 2))).place(x=450, y=240)
    r2_Svr = Label(top, text=str(round(r2_svr, 2))).place(x=450, y=300)
    current_value = Label(top, text=str(round(previous_close, 2))).place(x=600, y=210)
    top.mainloop()

def run():
    ticker = 'BTCUSDT'
    interval = '1d'
    limit = 1000
    window_size = 60
    data = get_binance_datarequest(ticker, interval, limit)
    X, y, previous_close, simulated_next_day = prepare_data(data, window_size)

    last_known_data = X[-1]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred_rf, r2_rf, mae_rf, predicted_rf = randomForestRegression(X_train, y_train, last_known_data)
    y_pred_lr, r2_lr, mae_lr, predicted_lr = linear_regression(X_train, y_train, last_known_data)
    y_pred_knn, r2_knn, mae_knn, predicted_knn = knn_regression(X_train, y_train, last_known_data)
    y_pred_svr, r2_svr, mae_svr, predicted_svr = svr_regression(X_train, y_train, last_known_data)
    y_pred_mlp, r2_mlp, mae_mlp, predicted_mlp = mlp_regression(X, y, last_known_data)

    design(predicted_rf, r2_rf, mae_rf, predicted_lr, r2_lr, mae_lr, predicted_knn, r2_knn, mae_knn, predicted_svr, r2_svr, mae_svr, previous_close)
run()
