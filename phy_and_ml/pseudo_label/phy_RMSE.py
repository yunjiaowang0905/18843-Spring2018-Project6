import sys
import numpy as np

phy = np.zeros((5942, 73, 18))


print('Load phy data')
with open(sys.argv[1], 'r') as fp:
	for line in fp:
		time,x,y,pm25 = line.strip().split(',')
		time = int(float(time))
		x = int(float(x))
		y = int(float(y))
		pm25 = float(pm25)
		phy[time][x][y] = pm25

print('Load Station data')

sqrt_sum = 0.0
total = 0

with open(sys.argv[2], 'r') as fp:
	for line in fp:
		time,x,y,pm25 = line.strip().split(',')
		time = int(float(time))
		x = int(float(x))
		y = int(float(y))
		pm25 = float(pm25)
		sqrt_sum += (pm25-phy[time][x][y])*(pm25-phy[time][x][y])
		total+=1

RMSE = np.sqrt(sqrt_sum/total)

print(RMSE)