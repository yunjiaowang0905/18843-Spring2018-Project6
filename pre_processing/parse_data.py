import scipy.io as sio
import geopy.distance
import numpy as np
import yaml
import pandas as pd
import glob
import matplotlib.pyplot as plt
import csv
from datetime import datetime, timedelta


def readFiles(path):
    filepath = []
    for fname in glob.glob(path):
        filepath.append(fname)
    return filepath

def filterPos(df, min_lat, max_lat, min_lon, max_lon):
    lat_mask = df[4] >= min_lat
    lat_mask &= df[4] <= max_lat
    lon_mask = df[5] >= min_lon
    lon_mask &= df[5] <= max_lon
    return df[lat_mask & lon_mask]

def filterTime(df, min_time, max_time):
    mask = df[0] >= min_time
    mask &= df[0] <= max_time
    print(df[~mask])
    return df[mask]

def filterPM2d5(df, min_pm2d5, max_pm2d5):
    mask = df[10] >= min_pm2d5
    mask &= df[10] <= max_pm2d5
    print(df[~mask])
    df[~mask].to_csv('filtered_pm2d5.csv', index = False)
    return df[mask]

def filterCO(df, min_co, max_co):
    mask = df[1] >= min_co
    mask &= df[1] <= max_co
    print(df[~mask])
    return df[mask]

def filterCO2(df, min_co2, max_co2):
    mask = df[2] >= min_co2
    mask &= df[2] <= max_co2
    print(df[~mask])
    return df[mask]

def filterNO2(df, min_no2, max_no2):
    mask = df[6] >= min_no2
    mask &= df[6] <= max_no2
    print(df[~mask])
    return df[mask]

def checkSpeedContinuity(df, max_speed):
    res = getSpeedByTimeLatLon(df)
    for i in range(res.shape):
        if res[i] > max_speed:
            print(df[i])

def getSpeedByTimeLatLon(df):
    res = np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        if i == 0:
            continue
        pre_lat = df[4].loc[i - 1]
        pre_lon = df[5].loc[i - 1]
        cur_lat = df[4].loc[i]
        cur_lon = df[5].loc[i]
        coords_1 = (cur_lat, cur_lon)
        coords_2 = (pre_lat, pre_lon)
        res[i] = geopy.distance.vincenty(coords_1, coords_2).km
    return res

def datetime2matlabdn(dt):
   mdn = dt + timedelta(days = 366)
   frac_seconds = (dt-datetime.datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
   frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
   return mdn.toordinal() + frac_seconds + frac_microseconds

def getPM2d5DiffsByCarID(path):
    car_id = path[0:path.find('.')]
    mat_contents = sio.loadmat(path)
    list_per_car = pd.DataFrame(mat_contents['DATA'])
    res = np.zeros(list_per_car.shape[0])
    # Sort by time
    list_per_car = list_per_car.sort_values(list_per_car.columns[0], ascending=True)
    for i in range(list_per_car.shape[0]):
        if i == 0:
            continue
        pre_pm2d5 = list_per_car[10].loc[i - 1]
        cur_pm2d5 = list_per_car[10].loc[i]
        res[i] = abs(cur_pm2d5 - pre_pm2d5)
    np.savetxt(car_id + '_diffs_pm2d5.csv', res, delimiter=",")
    return res

def checkPM2D5ChangeRangeByCarID(path, min_time_period, max_pm2d5_diff):
    car_id = path[0:path.find('.')]
    mat_contents = sio.loadmat(path)
    list_per_car = pd.DataFrame(mat_contents['DATA'])
    list_per_car = list_per_car.sort_values(list_per_car.columns[0], ascending=True)
    res = pd.DataFrame()
    for i in range(list_per_car.shape[0]):
        if i == 0:
            continue
        pre_time = list_per_car[4].loc[i - 1]
        pre_pm2d5 = list_per_car[5].loc[i - 1]
        cur_time = list_per_car[4].loc[i]
        cur_pm2d5 = list_per_car[5].loc[i]
        if cur_time - pre_time < min_time_period and abs(cur_pm2d5 - pre_pm2d5) > max_pm2d5_diff:
            print(list_per_car[i])
            res.append(list_per_car[i])
    res.to_csv(car_id + '_checkPM2D5Change.csv', index=False)
    return res

# Get parameters
params_fn = "parameters.yaml"
params_f = open(params_fn)
params = yaml.load(params_f)

# Read files
paths = readFiles('sensor_data/*.mat')
list_a = pd.DataFrame()
for path in paths:
    mat_contents = sio.loadmat(path)
    list_a = list_a.append(pd.DataFrame(mat_contents['DATA']))

print(len(list_a))

# Check ranges
list_a = filterPos(list_a, 18, 24, 112, 115)
list_a = filterPM2d5(list_a, float(params['Min_pm2d5']), float(params['Max_pm2d5']))
list_a = filterTime(list_a, float(params['Min_time']), float(params['Max_time']))
list_a = filterCO(list_a, float(params['Min_co']), float(params['Max_co']))
list_a = filterCO2(list_a, float(params['Min_co2']), float(params['Max_co2']))
list_a = filterNO2(list_a, float(params['Min_no2']), float(params['Max_no2']))

# Get pm2d5 diffs for each car
for path in paths:
    getPM2d5DiffsByCarID(path)

# Check pm2d5 change range for each car
for path in paths:
    checkPM2D5ChangeRangeByCarID(path, float(params['Min_time_period']), float(params['Max_pm2d5_diff']))

# Draw plots
max_lantitude = list_a[4].max()
max_longtitude = list_a[5].max()
min_lantitude = list_a[4].min()
min_longtitude = list_a[5].min()

print(max_lantitude)
print(max_longtitude)
print(min_lantitude)
print(min_longtitude)

plt.xlim(min_lantitude, max_lantitude)
plt.ylim(min_longtitude, max_longtitude)
plt.scatter(list_a[4],list_a[5])
plt.show()