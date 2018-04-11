import csv

readfile = '../preprocessing/classification_Res.csv'
writefile_train = 'timeData_Train.csv'
writefile_test = 'timeData_Test.csv'

headers = ['Day1', 'Day2', 'Day3', 'Today']

rf = open(readfile, 'rb')
wf_train = open(writefile_train, 'wb')
wf_test = open(writefile_test, 'wb')
reader = csv.DictReader(rf)
writer_train = csv.DictWriter(wf_train, headers)
writer_test = csv.DictWriter(wf_test, headers)

# for header in first line
writer_train.writerow({'Day1': 'Day1',
                       'Day2': 'Day2',
                       'Day3': 'Day3',
                       'Today': 'Today'})
writer_test.writerow({'Day1': 'Day1',
                      'Day2': 'Day2',
                      'Day3': 'Day3',
                      'Today': 'Today'})

# skip the first two lines
first_row = next(reader)
second_row = next(reader)
third_row = next(reader)

train_num = 1045393 / 10 * 9
cnt = 0

for row in reader:
    input_row = {'Day1': first_row['Level'],
                 'Day2': second_row['Level'],
                 'Day3': third_row['Level'],
                 'Today': row['Level']}
    first_row = second_row
    second_row = third_row
    third_row = row
    if cnt < train_num:
        writer_train.writerow(input_row)
    else:
        writer_test.writerow(input_row)
    cnt += 1

rf.close()
wf_train.close()
wf_test.close()