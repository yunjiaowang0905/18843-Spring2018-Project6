import sys
import numpy as np
from scipy.stats.stats import pearsonr

phy = np.zeros((5942, 73, 18))

data_ratio = {}
data_cor = {}
sqrt_sum = {}
total = {}
all_pm25 = {}
max_time = 0

print('Load phy data')
with open(sys.argv[1], 'r') as fp:
	for line in fp:
		time,x,y,pm25 = line.strip().split(',')
		time = int(float(time))
		x = int(float(x))
		y = int(float(y))
		pm25 = float(pm25)
		phy[time][x][y] = pm25

		max_time = max(max_time, time)

print('Load practical data')
with open(sys.argv[2], 'r') as fp:
	for line in fp:
		time,x,y,pm25 = line.strip().split(',')
		time = int(float(time))
		x = int(float(x))
		y = int(float(y))
		pm25 = float(pm25)

		if time not in data_ratio:
			data_ratio[time] = 0.0
			sqrt_sum[time] = 0.0
			total[time] = 0.0
			all_pm25[time] = 0.0
			data_cor[time] = set()

		if pm25 != -1.0 and (x,y) not in data_cor[time]:
			data_ratio[time]+=1
			data_cor[time].add((x,y))


print('Load Station data')

with open(sys.argv[3], 'r') as fp:
	for line in fp:
		time,x,y,pm25 = line.strip().split(',')
		time = int(float(time))
		x = int(float(x))
		y = int(float(y))
		pm25 = float(pm25)
		sqrt_sum[time] += (pm25-phy[time][x][y])*(pm25-phy[time][x][y])
		all_pm25[time] += pm25
		total[time]+=1

ratios = []
rmses = []
avgs = []

with open(sys.argv[4], 'w') as fp:
	for i in range(max_time+1):
		if total[i] > 0:
			ratio = data_ratio[i]/(73*18)
			RMSE = np.sqrt(sqrt_sum[i]/total[i])
			avg_pm25 = all_pm25[i]/total[i]
			print('Time: {}, Data Ratio: {}, RMSE: {}'.format(i, ratio, RMSE))
			fp.write(str(i)+','+str(ratio)+','+str(RMSE)+','+str(avg_pm25)+'\n')
			ratios.append(ratio)
			rmses.append(RMSE)
			avgs.append(avg_pm25)

print('Data Ratio x RMSE, Pearson correlation coefficient: {}'.format(pearsonr(ratios, rmses)))
print('AVG PM25 x RMSE, Pearson correlation coefficient: {}'.format(pearsonr(avgs, rmses)))