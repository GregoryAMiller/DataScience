import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn 

## read data into a data frame
data = pd.read_csv('C:/Users/Grego/Desktop/tkinter/Data1.csv')
## Prepare data
X = data.iloc[:, 0:3]     ## TC
Y = data.iloc[:, 4]     ## Idx
print(X)
print(Y)
print(X.shape)
print(Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = linear_model.LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f'coeficients: {model.coef_}')
# Y = 9.04182023e-04(T) + 1.16793126e-04(P) + -8.42308885e+00(TC) + intercept(53.9765861525509)
print(f'intercept {model.intercept_}')
print(f'mean squared error (MSE): {mean_squared_error(y_test, y_pred)}')
print(f'r2: {r2_score(y_test, y_pred)}')

# seaborn.scatterplot(x=y_test, y=y_pred)
plt.scatter(y_test, y_pred, alpha=.2, marker="+")
plt.show()
