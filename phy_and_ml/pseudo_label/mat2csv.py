import scipy.io
import sys
import numpy as np

mat = scipy.io.loadmat(sys.argv[1])

data = np.array(mat['data_interp_all'])
time, la, lo = data.shape

with open(sys.argv[2], 'w') as fp:
	for t in range(time):
		for x in range(la):
			for y in range(lo):
				fp.write(str(t)+','+str(x)+','+str(y)+','+str(data[t][x][y])+'\n')


