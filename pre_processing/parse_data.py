import scipy.io as sio
import geopy.distance
import numpy as np
import yaml
import pandas as pd
import glob
import matplotlib.pyplot as plt


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
    if mask is False:
        print(df[mask])
    return df[mask]

def filterPM2d5(df, min_pm2d5, max_pm2d5):
    mask = df[0] >= min_pm2d5
    mask &= df[0] <= max_pm2d5
    return df[mask]

def filterCO(df, min_co, max_co):
    mask = df[0] >= min_co
    mask &= df[0] <= max_co
    return df[mask]

def filterCO2(df, min_co2, max_co2):
    mask = df[0] >= min_co2
    mask &= df[0] <= max_co2
    return df[mask]

def filterNO2(df, min_no2, max_no2):
    mask = df[0] >= min_no2
    mask &= df[0] <= max_no2
    return df[mask]


# Get parameters
params_fn = "parameters.yaml"
params_f = open(params_fn)
params = yaml.load(params_f)
params['Max_time'] = float(params['Max_time'])
params['Min_time'] = float(params['Min_time'])

# Read files
paths = readFiles('../data/sensor_data/*.mat')
list_a = pd.DataFrame()
for path in paths:
    mat_contents = sio.loadmat(path)
    list_a = list_a.append(pd.DataFrame(mat_contents['DATA']))

# Check ranges
list_a = filterPos(list_a, 18, 24, 112, 115)

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