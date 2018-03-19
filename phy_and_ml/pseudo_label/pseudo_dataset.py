import numpy as np
import sys
import random

origin = np.zeros((5942, 73, 18))
phy = np.zeros((5942, 73, 18))

cnt = 0
total =0

print('Load original data')
with open(sys.argv[1], 'r') as fp:
	for line in fp:
		time,x,y,pm25 = line.strip().split(',')
		time = int(float(time))
		x = int(float(x))
		y = int(float(y))
		pm25 = float(pm25)
		origin[time][x][y] = pm25
		if pm25 >= 0:
			cnt+=1
		total+=1
print('Load physical predicted data')
with open(sys.argv[2], 'r') as fp:
	for line in fp:
		time,x,y,pm25 = line.strip().split(',')
		time = int(float(time))
		x = int(float(x))
		y = int(float(y))
		pm25 = float(pm25)
		phy[time][x][y] = pm25

rate = (float(cnt)/total)*float(sys.argv[4])

with open(sys.argv[3], 'w') as fp:
	for i in range(5942):
		for j in range(73):
			for k in range(18):
				if origin[i][j][k] >=0:
					fp.write(str(i)+','+str(j)+','+str(k)+','+str(origin[i][j][k])+'\n')
				else:
					if random.random() > rate:
						fp.write(str(i)+','+str(j)+','+str(k)+','+str(origin[i][j][k])+'\n')
					else:
						fp.write(str(i)+','+str(j)+','+str(k)+','+str(phy[i][j][k])+'\n')