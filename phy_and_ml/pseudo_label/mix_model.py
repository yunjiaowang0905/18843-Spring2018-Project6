import sys
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np

def RMSE(y_pred, y):
	return np.sqrt(np.mean(np.power(y_pred - y, 2)))

X_train, y_train = load_svmlight_file(sys.argv[1])
X_train = preprocessing.scale(X_train.toarray())
X_test, y_test = load_svmlight_file(sys.argv[2])
X_test = preprocessing.scale(X_test.toarray())

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

phy = np.zeros((5942, 73, 18))
all_pm25 = {}
total = {}

with open(sys.argv[3], 'r') as fp:
	for line in fp:
		time,x,y,pm25 = line.strip().split(',')
		time = int(float(time))
		x = int(float(x))
		y = int(float(y))
		pm25 = float(pm25)
		phy[time][x][y] = pm25
		all_pm25[time] = 0.0
		total[time] = 0


with open(sys.argv[4], 'r') as fp:
	for line in fp:
		time,x,y,pm25 = line.strip().split(',')
		time = int(float(time))
		x = int(float(x))
		y = int(float(y))
		pm25 = float(pm25)
		if pm25 != -1.0:
			all_pm25[time] += pm25
			total[time] +=1


threshold = float(sys.argv[6])
cnt = 0
with open(sys.argv[5], 'r') as fp:
	for line in fp:
		time,x,y,pm25 = line.strip().split(',')
		time = int(float(time))
		x = int(float(x))
		y = int(float(y))
		pm25 = float(pm25)
		if total[time] == 0:
			cnt+=1
			continue

		if all_pm25[time]/total[time] < threshold:
			y_pred[cnt] = phy[time][x][y]
		cnt+=1

rmse = RMSE(y_pred, y_test)
print(rmse)