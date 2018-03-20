__author__ = 'Jin Gong'
import numpy as np
import scipy.io
from scipy.interpolate import interp1d
from math import pi
from datetime import datetime, timedelta

def get_pre_slt(file_path, date_min, date_max, gas_max_val, car_no, lon_min, lon_max, lat_min, lat_max, n_lat, n_lon,t_min, t_max):
    """
    select data according to given date, time period, car id, area range
    output:
    data_slt: selected data
    xi,yi: longitude and latitude seperation points
    """
    data_mat = scipy.io.loadmat(file_path)['raw_data_area_time']
    data_t_idx = np.nonzero((data_mat[:,1] >= date_min)\
        & (data_mat[:,1] <= date_max)\
        & (data_mat[:,4] > 0)\
        & (data_mat[:,4] < gas_max_val)\
        & (np.in1d(data_mat[:,0], range(1, car_no + 1)))) # car selection
    data_t = data_mat[data_t_idx[0],:]
    # area selection
    data_q_idx = np.nonzero((data_t[:,2] >= lat_min)\
        & (data_t[:,2] <= lat_max)\
        & (data_t[:,3] >= lon_min)\
        & (data_t[:,3] <= lon_max))
    data_q = data_t[data_q_idx[0],:]

    xi = np.linspace(lon_min, lon_max, num=n_lon)
    yi = np.linspace(lat_min, lat_max, num=n_lat)
    #TODO:test
    xr = interp1d(xi, range(1, n_lon + 1), kind="nearest")(data_q[:,3])
    yr = interp1d(yi, range(1, n_lat + 1), kind="nearest")(data_q[:,2])
    data_q = np.c_[data_q, data_q[:,2], data_q[:,3]] # append two columns
    data_q[:,2] = xr
    data_q[:,3] = yr
    d_lon = lon_max - lon_min
    d_lat = d_lon * np.cos(lat_min / 180 * pi) * n_lat / n_lon
    xi = np.linspace(lon_min + d_lon/n_lon/2, lon_max - d_lon/n_lon/2, num=n_lon)
    yi = np.linspace(lat_min + d_lat/n_lat/2, lat_max - d_lat/n_lat/2, num=n_lat)
    for data in data_q:
        hour = (datetime.fromordinal(int(date[1])) + timedelta(days=data[1]%1) - timedelta(days = 366)).hour
        if hour >= t_min and hour < t_max:
            data_slt.append(data)
    return xi, yi, data_slt
