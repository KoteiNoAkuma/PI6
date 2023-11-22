import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import sklearn

def calculateVolume(data):
    return data['volume']

# # MLP Regression
# def mlp_regression(X_train, X_test, y_train, y_test):
#     mlp_regressor = MLPRegressor(random_state=1, max_iter=1000)
#     mlp_regressor.fit(X_train, y_train)
#     y_pred = mlp_regressor.predict(X_test)

#     mse = mean_squared_error(y_test, y_pred)
#     print('===========')
#     print("Real Data: ")
#     print(y_test)
#     print('===========')
#     print("Predicted Data: ")
#     print(y_pred)
#     print('===========')

#     return y_pred, mse

# y_pred_mlp, mse_mlp = mlp_regression(X_train, X_test, y_train, y_test)
# print("MLP MSE: ", mse_mlp)

# print(history_df) # Retorna dados de 16 de março de 2022 até 2 de novembro de 2023  597 dias  (Com algumas lacunas)
