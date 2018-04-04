import numpy as np
import sys

fp1 = open(sys.argv[1], 'r')
fp2 = open(sys.argv[2], 'w')

for line in fp1:
	time, x, y, pm25 = line.strip().split(',')
	pm25 = float(pm25)
	if np.isnan(pm25):
		continue
	fp2.write(line)
