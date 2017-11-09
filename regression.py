'''
Revised by Bryan, 11.09,2017
This is a code realizing the linearRegression
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('F:\machineLearning\house_data.csv')

# cols = ['LSTAT', 'AGE', 'DIS', 'CRIM', 'MEDV', 'TAX', 'RM']
# sns.pairplot(df[cols], size=2.5)
# plt.show()

class LinearRegressionByMySelf(object):
    def __init__(self, learning_rate = 0.001, epoch = 20):
        self.learning_rate = learning_rate
        self.epoch = epoch

    def fit(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        self.cost_list = []

        for i in range(self.epoch):
            output = self.Regression_input(X)
            error = (y - output)
            self.w[1:] += self.learning_rate * X.T.dot(error)
            self.w[0] += self.learning_rate * error.sum()
            cost = (error ** 2).sum() / 2.0
            self.cost_list.append(cost)
        return self

    def Regression_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        return self.Regression_input(X)


X = df[['LSTAT']].values
y = df['MEDV'].values

StandardScaler_x = StandardScaler()
StandardScaler_y = StandardScaler()

X_Standard = StandardScaler_x.fit_transform(X)
y_Standard = StandardScaler_y.fit_transform(y.reshape(-1,1))

model = LinearRegressionByMySelf()
model.fit(X_Standard, y_Standard.ravel())

plt.plot(range(1, model.epoch+1), model.cost_list)
plt.ylabel('SSE')
plt.xlabel('Epoch')

plt.show()