import sys

hor = 73/2
ver = 18/2

type = sys.argv[4]

with open (sys.argv[1], 'r') as fp:
	fp2 = open(sys.argv[2], 'w')
	fp3 = open(sys.argv[3], 'w')

	for line in fp:
		tmp = line.strip().split(',')
		x = float(tmp[1])
		y = float(tmp[2])
		value = float(tmp[3])

		if value == -1.0:
			fp2.write(line)
			continue

		if type == 'h':
			if y < ver:
				fp2.write(line)
			else:
				tmp[3] = '-1.0'
				fp2.write(','.join(tmp)+'\n')
				fp3.write(line)
		elif type == 'v':
			if x < hor:
				fp2.write(line)
			else:
				tmp[3] = '-1.0'
				fp2.write(','.join(tmp)+'\n')
				fp3.write(line)