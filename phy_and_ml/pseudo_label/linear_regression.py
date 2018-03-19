import sys
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LinearRegression
import numpy as np

def RMSE(y_pred, y):
	return np.sqrt(np.mean(np.power(y_pred - y, 2)))

X_train, y_train = load_svmlight_file(sys.argv[1])
X_test, y_test = load_svmlight_file(sys.argv[2])

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

rmse = RMSE(y_pred, y_test)
print(rmse)