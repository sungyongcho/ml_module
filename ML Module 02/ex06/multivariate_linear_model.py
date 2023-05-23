

import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR
data = pd.read_csv("spacecraft_data.csv")
X = np.array(data[['Age']])
Y = np.array(data[['Sell_price']])
myLR_age = MyLR(thetas=[[1000.0], [-1.0]], alpha=2.5e-5, max_iter=100000)
myLR_age.fit_(X[:, 0].reshape(-1, 1), Y)
y_pred = myLR_age.predict_(X[:, 0].reshape(-1, 1))
# print(y_pred)
print(myLR_age.mse_(y_pred, Y))
# Output
# 55736.86719...
myLR_age.plot_regression(X, Y, y_pred, 'Age')

X = np.array(data[['Thrust_power']])
Y = np.array(data[['Sell_price']])
myLR_thrust = MyLR(thetas=[[1000.0], [-1.0]], alpha=2.5e-5, max_iter=100000)
myLR_thrust.fit_(X[:, 0].reshape(-1, 1), Y)
y_pred = myLR_thrust.predict_(X[:, 0].reshape(-1, 1))
# print(y_pred)
print(myLR_thrust.mse_(y_pred, Y))
myLR_thrust.plot_regression(X, Y, y_pred, 'Thrust_power')


X = np.array(data[['Terameters']])
Y = np.array(data[['Sell_price']])
myLR_distance = MyLR(thetas=[[1000.0], [-1.0]], alpha=2.5e-5, max_iter=100000)
myLR_distance.fit_(X[:, 0].reshape(-1, 1), Y)
y_pred = myLR_distance.predict_(X[:, 0].reshape(-1, 1))
# print(y_pred)
print(myLR_distance.mse_(y_pred, Y))
myLR_distance.plot_regression(X, Y, y_pred, 'Terameters')

X = np.array(data[['Age', 'Thrust_power', 'Terameters']])
Y = np.array(data[['Sell_price']])
thetas = np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
my_lreg = MyLR(thetas, alpha=1e-5, max_iter=500000)
print("mse:", my_lreg.mse_(X, Y), "\n")
print(np.array(data[['Age']]))

# # Example 1:
my_lreg.fit_(X, Y)
y_pred = my_lreg.predict_(X)
# # # Output:
# # 144044.877...
# # Example 1:
my_lreg.fit_(X, Y)
print(my_lreg.thetas)
# # Output:
# array([[334.994...],[-22.535...],[5.857...],[-2.586...]])

# Example 2:
print(my_lreg.mse_(X, Y))
# # # # Output:
# # # 586.896999...
# column_names = ['Age', 'Thrust_power', 'Terameters']

col = 'Age'
if col == 'Age':
    x_np = np.array(data[['Age']])
elif col == 'Thrust_power':
    x_np = np.array(data[['Thrust_power']])
elif col == 'Terameters':
    x_np = np.array(data[['Terameters']])

my_lreg.plot_multivariable_regression(x_np, Y, y_pred, col)

col = 'Thrust_power'
if col == 'Age':
    x_np = np.array(data[['Age']])
elif col == 'Thrust_power':
    x_np = np.array(data[['Thrust_power']])
elif col == 'Terameters':
    x_np = np.array(data[['Terameters']])

my_lreg.plot_multivariable_regression(x_np, Y, y_pred, col)


col = 'Terameters'
if col == 'Age':
    x_np = np.array(data[['Age']])
elif col == 'Thrust_power':
    x_np = np.array(data[['Thrust_power']])
elif col == 'Terameters':
    x_np = np.array(data[['Terameters']])

my_lreg.plot_multivariable_regression(x_np, Y, y_pred, col)
