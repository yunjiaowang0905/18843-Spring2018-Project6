__author__ = 'Nanshu Wang'

import numpy as np
import sys, os
import scipy.io
from scipy.interpolate import interp1d
from math import pi
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(".."))

from util import rcd2grid

class baselines(object):
    """Baselines
    This class offers the data seperation
    Parameters
    """
    def __init__(self, conf, res_t, flag_empty, date_min, date_max, n_lat, n_lon, data):
        """
        :param conf: configure for particle filter, please refer conf/default.yml
        :param n_lat: latitude grids number
        :param n_lon: longitude grids number
        :param data: data_schema class, please refer util/data_helper.py
        """

        self.flag_empty = flag_empty
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.data = data

        self.res_t = conf['rest_t']
        self.lon_min = conf['lon_min']
        self.lon_max = conf['lon_max']
        self.lat_min = conf['lat_min']
        self.lat_max = conf['lat_max']
        self.t_min = conf['t_min']
        self.t_max = conf['t_max']
        self.date_min_num = date_min.toordinal() + 366
        self.date_max_num = date_max.toordinal() + 366

        self.car_no = conf['car_no']
        self.gas_max_val = conf['gas_max_val']
        self.pre_max_val = 10

        self.idx_station_v = None
        self.data_slt = None
        self.station_data_slt = None

        self.data_selection()


    def get_pre_slt(self, file_path):
        """
        select data according to given date, time period, car id, area range
        data_slt: selected data
        xi,yi: longitude and latitude seperation points
        """
        data_mat = scipy.io.loadmat(file_path)['raw_data_area_time']
        data_t_idx = np.nonzero((data_mat[:,1] >= self.date_min_num)\
            & (data_mat[:,1] <= self.date_min_num)\
            & (data_mat[:,4] > 0)\
            & (data_mat[:,4] < self.gas_max_val)\
            & (np.in1d(data_mat[:,0], range(1, self.car_no + 1)))) # car selection
        data_t = data_mat[data_t_idx[0],:]
        # area selection
        data_q_idx = np.nonzero((data_t[:,2] >= self.lat_min)\
            & (data_t[:,2] <= self.lat_max)\
            & (data_t[:,3] >= self.lon_min)\
            & (data_t[:,3] <= self.lon_max))
        data_q = data_t[data_q_idx[0],:]

        xi = np.linspace(self.lon_min, self.lon_max, num=self.n_lon)
        yi = np.linspace(self.lat_min, self.lat_max, num=self.n_lat)
        xr = interp1d(xi, range(1, self.n_lon + 1), kind="nearest")(data_q[:,3])
        yr = interp1d(yi, range(1, self.n_lat + 1), kind="nearest")(data_q[:,2])
        data_q = np.c_[data_q, data_q[:,2], data_q[:,3]] # append two columns
        data_q[:,2] = xr
        data_q[:,3] = yr
        d_lon = self.lon_max - self.lon_min
        d_lat = d_lon * np.cos(self.slat_min / 180 * pi) * self.n_lat / self.n_lon
        xi = np.linspace(self.lon_min + d_lon/self.n_lon/2, self.lon_max - d_lon/self.n_lon/2, num=self.n_lon)
        yi = np.linspace(self.lat_min + d_lat/self.n_lat/2, self.lat_max - d_lat/self.n_lat/2, num=self.n_lat)
        data_slt_list = []
        for d in data_q:
            hour = (datetime.fromordinal(int(d[1])) + timedelta(days=d[1]%1) - timedelta(days = 366)).hour
            if hour >= self.t_min and hour < self.t_max:
                data_slt_list.append(d)
        return xi, yi, data_slt_list


    def data_selection(self):
        xi, yi, data_slt_list = data_pre_slt(DATA_DIR+ "/201701/raw/slt_raw_area_time_early.mat"）
        self.data_slt = np.asarray(data_slt_list)
        # get the data from monitoring station
        station_info = scipy.io.loadmat(DATA_DIR + "/data/station_info.mat")
        station_data = scipy.io.loadmat(DATA_DIR + "/data/station_data.mat")
        station_info_lon = np.asarray([v[1] for v in station_info['station_info']]).flatten()
        station_info_lat = np.asarray([v[2] for v in station_info['station_info']]).flatten()
        station_loc = np.c_[station_info_lon, station_info_lat]
        station_loc_idx = np.nonzero((station_loc[:,0] >= self.lon_min) & (station_loc[:, 0] <= self.lon_max) \
            & (station_loc[:,1] >= self.lat_min) & (station_loc[:, 1] <= self.lat_max))
        station_data_loc = station_data['station_data'][:, np.r_[0, station_loc_idx[0] + 2]]
        station_data_idx = np.nonzero((station_data_loc[:,0] >= date_min_num) & (station_data_loc[:,0] <= date_max_num))
        self.station_data_slt = station_data_loc[station_data_idx[0],:]
        #pre_max_val = np.max(station_data_slt[:,1:len(station_data_slt[0])])
        self.idx_station_v = np.zeros(np.shape(station_loc_idx)[1])
        #idx_sta = np.zeros(self.n_lat, self.n_lon) # n_lon=64, n_lat=16
        #sta_loc = np.zeros((np.shape(station_loc_idx)[1], 2))

        for i in range(0, np.shape(station_loc_idx)[1]):
            loc_cur = station_loc[station_loc_idx[0][i],:]
            # a = min(abs(loc_cur[0] - xi))
            # b = min(abs(loc_cur[1] - yi))
            x_cur = np.argmin(abs(loc_cur[0] - xi))
            y_cur = np.argmin(abs(loc_cur[1] - yi))
            self.idx_station_v[i]= x_cur * n_lat + (y_cur + 1)
            #idx_sta[y_cur, x_cur] = 1
            #sta_loc[i,:] = [y_cur, x_cur]

    def run_iter(self, i_t):
        t_low = self.date_min_num + i_t * self.res_t / 24
        t_high = self.date_min_num + (i_t + 1) * self.res_t / 24 #TODO: not data_max_num???
        # dispaly(datastr(t_low))
        data_cur_idx = np.nonzero((self.data_slt[:, 1] >= t_low) & (self.data_slt[:, 1] < t_high))
        data_cur = self.data_slt[data_cur_idx[0],:]
        delt = 10 / 24 / 60
        data_station_cur_idx = np.nonzero((self.station_data_slt[:, 0] >= (t_low - delt))\
            & (self.station_data_slt[:, 0] < (t_high - delt)))
        datadata_station_cur = self.station_data_slt[data_station_cur_idx[0],:]
        if data_cur.size:
            self.flag_empty = False
        else:
            self.flag_empty = True

        #TODO: test
        baseML(data_cur, i_t)

        # evaluation
        # eva_re_err_dd: i_t * 16 * 64
        self.data.eva_re_err_dd[i_t,:,:] = get_relative_error(self.data.x_est_dd[i_t], self.data.data_gt[i_t], self.data.smp_cnt_gt[i_t],1)
        self.data.eva_re_err_ann[i_t] = get_relative_error(self.data.x_est_ann[i_t], self.data.data_gt[i_t], self.data.smp_cnt_gt[i_t],1)
        self.data.eva_re_err_gp[i_t] = get_relative_error(self.data.x_est_gp[i_t], self.data.data_gt[i_t], self.data.smp_cnt_gt[i_t],1)

        # use station data to evaluation
        if data_station_cur.size
            ds_cur = data_station_cur[1,:]
            ds_cur[ds_cur is 0] = 1e-4;
            self.daya.eva_all(i_t,:,:)=abs([self.data.x_est_dd[i_t](self.idx_station_v), self.data.x_est_ann[i_t](self.idx_station_v)\
                self.data.x_est_gp[i_t](self.idx_station_v)]\
                -repmat(ds_cur',1,3))./repmat(ds_cur',1,3);
            self.data.data_station_all[i_t,:,:] = ds_cur;
