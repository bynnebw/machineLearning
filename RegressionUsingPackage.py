# -*- coding: utf-8 -*-

"""
使用sklearn包进行线性回归
并进行一些相关的评价与验证工作
"""

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('F:\machineLearning\house_data.csv')

X = df[['LSTAT']].values
y = df['MEDV'].values


sk_model = LinearRegression()
sk_model.fit(X, y)

def Regression_plot(X,y,model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

# Regression_plot(X,y,sk_model)
# plt.xlabel('Percentage of the populations')
# plt.ylabel('House price')
# plt.show()

# The next block of codes represents the part of drawing the scatter
# estimating the effeciency

cols = ['LSTAT', 'AGE', 'DIS', 'CRIM', 'TAX', 'RM']
X = df[cols].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sk_model.fit(X_train, y_train)
y_train_predict = sk_model.predict(X_train)
y_test_predict = sk_model.predict(X_test)

plt.scatter(y_train_predict, y_train_predict-y_train, c='red', marker='x', label='Trainning data')
plt.scatter(y_test_predict, y_test_predict-y_test, c='black', marker='o', label='Test data')

plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=1, colors='green')
plt.xlim([-10, 50])
plt.show()

print('MSE train %.3f, test %.3f ' %
      (mean_squared_error(y_train, y_train_predict),
       mean_squared_error(y_test, y_test_predict)))

print('R^2 train %.3f, test %.3f ' %
      (r2_score(y_train, y_train_predict),
       r2_score(y_test, y_test_predict)))

