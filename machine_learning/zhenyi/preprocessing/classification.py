import csv

readfile = '../distribution_Res.csv'
writefile = 'classification_Res.csv'

headers = ['Time', 'Longitude', 'Latitude', 'Pollution', 'AQI', 'Level']

rf = open(readfile, 'rb')
wf = open(writefile, 'wb')
reader = csv.DictReader(rf)
writer = csv.DictWriter(wf, headers)

level0 = 0
level1 = 0
level2 = 0
level3 = 0
level4 = 0
level5 = 0
tt = 0

writer.writerow({'Time': 'Time',
                 'Longitude': 'Longitude',
                 'Latitude': 'Latitude',
                 'Pollution': 'Pollution',
                 'AQI': 'AQI',
                 'Level': 'Level'})

for row in reader:
    pollution = float(row['Pollution'])
    if pollution != -1.0:
        aqi = 0
        if pollution >= 351:
            aqi = 100 * (pollution - 351) / 150 + 401
        elif pollution >= 251:
            aqi = 100 * (pollution - 251) / 100 + 301
        elif pollution >= 151:
            aqi = 100 * (pollution - 151) / 100 + 201
        elif pollution >= 116:
            aqi = 50 * (pollution - 116) / 35 + 151
        elif pollution >= 76:
            aqi = 50 * (pollution - 76) / 40 + 101
        elif pollution >= 36:
            aqi = 50 * (pollution - 36) / 40 + 51
        else:
            aqi = 50 * pollution / 35

        level = 0
        if aqi > 300:
            level = 5
            level5 += 1
        elif aqi > 200:
            level = 4
            level4 += 1
        elif aqi > 150:
            level = 3
            level3 += 1
        elif aqi > 100:
            level = 2
            level2 += 1
        elif aqi > 50:
            level = 1
            level1 += 1
        else:
            level0 += 1
        input_row = {'Time': row['Time'],
                     'Longitude': row['Longitude'],
                     'Latitude': row['Latitude'],
                     'Pollution': row['Pollution'],
                     'AQI': str(aqi),
                     'Level': str(level)}
        writer.writerow(input_row)

tt = (level0 + level1 + level2 + level3 + level4 + level5) * 1.0
print("level0: ", level0, level0 / tt)
print("level1: ", level1, level1 / tt)
print("level2: ", level2, level2 / tt)
print("level3: ", level3, level3 / tt)
print("level4: ", level4, level4 / tt)
print("level5: ", level5, level5 / tt)

rf.close()
wf.close()