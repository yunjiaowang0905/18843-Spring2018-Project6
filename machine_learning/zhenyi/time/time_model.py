import csv
from sklearn import tree

readfile_train = 'timeData_Train.csv'
readfile_test = 'timeData_Test.csv'

rf_train = open(readfile_train, 'rb')
rf_test = open(readfile_test, 'rb')
reader_train = csv.DictReader(rf_train)
reader_test = csv.DictReader(rf_test)

x_train = []
y_train = []

x_test = []
y_test_predict = []
y_test_real = []

for row in reader_train:
    cur_x_train = [int(row['Day1']), int(row['Day2']), int(row['Day3'])]
    x_train.append(cur_x_train)
    y_train.append(int(row['Today']))
rf_train.close()

for row in reader_test:
    cur_x_test = [int(row['Day1']), int(row['Day2']), int(row['Day3'])]
    x_test.append(cur_x_test)
    y_test_real.append(int(row['Today']))
rf_test.close()

# training
model = tree.DecisionTreeClassifier()
model = model.fit(x_train, y_train)

# testing
y_test_predict = model.predict(x_test)

# print result
level_tt = [0, 0, 0, 0, 0, 0]
level_true = [0, 0, 0, 0, 0, 0]

# confusion matrix
# row - actual
# column - predict
confusion_matrix = [[0 for i in range(6)] for j in range(6)]

for i in range(len(y_test_real)):
    level_tt[y_test_real[i]] += 1
    if y_test_real[i] == y_test_predict[i]:
        level_true[y_test_real[i]] += 1
    confusion_matrix[y_test_real[i]][y_test_predict[i]] += 1

for i in range(len(level_tt)):
    print(level_true[i], level_tt[i], level_true[i] * 1.0 / level_tt[i])

print confusion_matrix