import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

data = dict()

with open(input_file, 'r') as fp:
	for line in fp:
		time, x, y, pm25 = line.strip().split(',')
		if (x,y) not in data:
			data[(x,y)] = []
		if float(pm25) >= 0:
			data[(x,y)].append((time, pm25))

features = []

for cord in data:
	x, y = cord
	for i in range(len(data[cord])):
		time, pm25 = data[cord][i]
		time = int(float(time))
		if i == 0:
			previous = '-1'
		else:
			previous = data[cord][i-1][1]
		hour = str(time %24)
		weekday = str(int(time/24)%7)
		features.append((time, pm25, x, y, previous, hour, weekday))

features.sort(key=lambda tup: tup[0])

max_time = features[-1][0]

fp_train = open(sys.argv[2], 'w')
fp_test = open(sys.argv[3], 'w')

for feature in features:
	if feature[0] < max_time*0.9:
		fp = fp_train
	else:
		fp = fp_test
	fp.write(feature[1])
	fp.write(" 1:"+feature[2])
	fp.write(" 2:"+feature[3])
	fp.write(" 3:"+feature[4])
	fp.write(" 4:"+feature[5])
	fp.write(" 5:"+feature[6]+'\n')