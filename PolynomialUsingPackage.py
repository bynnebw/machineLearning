# -*- coding: utf-8 -*-

'''
多项式拟合，建模房屋数据集里的非线性关系
'''

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('F:\machineLearning\house_data.csv')

X = df[['RM']].values
y = df['MEDV'].values
Regression_model = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

X_squared = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# What's the actual meaning of the code?
X_fit = np.arange(X.min(), X.max(), 0.01)[:,np.newaxis]

Linear_model = Regression_model.fit(X,y)
y_line_fit = Linear_model.predict(X_fit)
linear_r2 = r2_score(y, Linear_model.predict(X))

Squared_model = Regression_model.fit(X_squared,y)
y_quad_fit = Squared_model.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, Squared_model.predict(X_squared))

Cubic_model = Regression_model.fit(X_cubic, y)
y_cubic_fit = Cubic_model.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, Cubic_model.predict(X_cubic))

plt.scatter(X, y, label='Training point', color='lightgray')
plt.plot(X_fit, y_line_fit, label='linear, $R^2=%.2f$' % linear_r2, color='blue', lw=2, linestyle=':')
plt.plot(X_fit, y_quad_fit, label='quadratic, $R^2=%.2f$' % quadratic_r2, color='red', lw=2, linestyle='-')
plt.plot(X_fit, y_cubic_fit, label='cubic, $R^2=%.2f$' % cubic_r2, color='green', lw=2, linestyle='--')

plt.xlabel('Room numbers')
plt.ylabel('House price')
plt.legend(loc='upper left')
plt.show()
