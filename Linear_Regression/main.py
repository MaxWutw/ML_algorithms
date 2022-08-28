## Finding the curve that best fits your data is called regression, and when that curve is a straight line, it's called linear regression.

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from linear_regression import LinearRegression

def MSE(y_true, predict):
    return np.mean((y_true - predict)**2)

x, y = datasets.make_regression(n_samples=100, n_features=1, noise=25, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=456)

regressor = LinearRegression(lr=0.01) # instances
regressor.fit(x_train, y_train)

predict = regressor.predict(x_test)

loss = MSE(y_test, predict)

print(loss)

plt.figure(figsize=(8, 6))
plt.scatter(x_train[:, 0], y_train, color="r", marker="o", s=20)
plt.scatter(x_test[:, 0], y_test, color="r", marker="o", s=20)
plt.plot(x_test, predict, color="black", linewidth=2, label="Prediction")
plt.show()