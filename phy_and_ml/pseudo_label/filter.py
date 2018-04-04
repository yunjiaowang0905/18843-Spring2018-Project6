import sys

threshold = float(sys.argv[3])

good_time = set()

with open(sys.argv[1], 'r') as fp:
	for line in fp:
		line = line.strip().split(",")
		data_ratio = float(line[1])
		RMSE = float(line[2])

		if RMSE <= threshold:
			good_time.add(int(line[0]))

total_pm25 = 0.0
count = 0

with open(sys.argv[2], 'r') as fp:
	for line in fp:
		line = line.strip().split(",")
		pm25 = float(line[3])
		time = int(float(line[0]))

		if time in good_time:
			total_pm25 += pm25
			count+=1

print(total_pm25/count, count)