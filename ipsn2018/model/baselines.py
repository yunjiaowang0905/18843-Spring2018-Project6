__author__ = 'Nanshu Wang'

import numpy as np
import pickle
import csv
import sys, os
import scipy.io
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from collections import defaultdict
from math import pi
from datetime import datetime, timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
sys.path.insert(0, os.path.abspath(".."))

from util import rcd2grid, limit_range, get_dd_result
from util.get_relative_error import get_relative_error
from util.get_dd_result_new_data import get_dd_result_new_data
DATA_DIR = "/Users/melody/18843-Spring2018-Project6/ipsn2018/data"

class baselines(object):
    """Baselines
    This class offers the data seperation
    Parameters
    """
    def __init__(self, conf, flag_empty, pct_mat, date_min, date_max, n_lat, n_lon, data):
        """
        :param conf: configure for particle filter, please refer conf/default.yml
        :param n_lat: latitude grids number
        :param n_lon: longitude grids number
        :param data: data_schema class, please refer util/data_helper.py
        """

        self.flag_empty = flag_empty
        self.pct_mat = pct_mat
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.data = data

        self.flag_range_lim =   conf['flag_range_lim']
        self.pct_gt = conf['pct_gt']
        self.t_len_his = conf['t_len_his']
        self.res_t = conf['res_t']
        self.alg_sep = conf['alg_sep']
        self.lon_min = conf['lon_min']
        self.lon_max = conf['lon_max']
        self.lat_min = conf['lat_min']
        self.lat_max = conf['lat_max']
        self.t_min = conf['t_min']
        self.t_max = conf['t_max']
        self.car_no = conf['car_no']
        self.gas_max_val = conf['gas_max_val']
        self.date_min_num = date_min.toordinal() + 366
        self.date_max_num = date_max.toordinal() + 366
        print "date_min_num" + str(self.date_min_num)
        print "date_max_num" + str(self.date_max_num)

        self.pre_max_val = 10
        self.idx_station_v = None
        self.data_slt = None
        self.station_data_slt = None

        #self.data_selection()
        self.data_pre_sep()


    def data_pre_slt(self, file_path):
        """
        select data according to given date, time period, car id, area range
        data_slt: selected data
        xi,yi: longitude and latitude seperation points
        """
        data_mat = scipy.io.loadmat(file_path)['raw_data_area_time']
        data_t_idx = np.nonzero((data_mat[:,1] >= self.date_min_num)\
            & (data_mat[:,1] <= self.date_max_num)\
            & (data_mat[:,4] > 0)\
            & (data_mat[:,4] < self.gas_max_val)\
            & (np.in1d(data_mat[:,0], range(1, self.car_no + 1)))) # car selection
        data_t = data_mat[data_t_idx[0],:]
        print "data_t"
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
        d_lat = d_lon * np.cos(self.lat_min / 180 * pi) * self.n_lat / self.n_lon
        xi = np.linspace(self.lon_min + d_lon/self.n_lon/2, self.lon_max - d_lon/self.n_lon/2, num=self.n_lon)
        yi = np.linspace(self.lat_min + d_lat/self.n_lat/2, self.lat_max - d_lat/self.n_lat/2, num=self.n_lat)
        data_slt_list = []
        print "data_q"
        for d in data_q:
            hour = (datetime.fromordinal(int(d[1])) + timedelta(days=d[1]%1) - timedelta(days = 366)).hour
            if hour >= self.t_min and hour < self.t_max:
                data_slt_list.append(d)
        return xi, yi, data_slt_list


    def data_selection(self):
        xi, yi, data_slt_list = self.data_pre_slt(DATA_DIR + "/201701/raw/slt_raw_area_time_early.mat")
        self.data_slt = np.asarray(data_slt_list)
        print np.shape(self.data_slt)
        # get the data from monitoring station
        station_info = scipy.io.loadmat(DATA_DIR + "/station_info.mat")
        station_data = scipy.io.loadmat( DATA_DIR + "/station_data.mat")
        print "Successfully load data"
        station_info_lon = np.asarray([v[1] for v in station_info['station_info']]).flatten()
        station_info_lat = np.asarray([v[2] for v in station_info['station_info']]).flatten()
        station_loc = np.c_[station_info_lon, station_info_lat]
        station_loc_idx = np.nonzero((station_loc[:,0] >= self.lon_min) & (station_loc[:, 0] <= self.lon_max) \
            & (station_loc[:,1] >= self.lat_min) & (station_loc[:, 1] <= self.lat_max))
        station_data_loc = station_data['station_data'][:, np.r_[0, station_loc_idx[0] + 2]]
        station_data_idx = np.nonzero((station_data_loc[:,0] >= self.date_min_num) & (station_data_loc[:,0] <= self.date_max_num))
        self.station_data_slt = station_data_loc[station_data_idx[0],:]
        #pre_max_val = np.max(station_data_slt[:,1:len(station_data_slt[0])])
        self.idx_station_v = np.zeros(np.shape(station_loc_idx)[1])
        #idx_sta = np.zeros(self.n_lat, self.n_lon) # n_lon=64, n_lat=16
        #sta_loc = np.zeros((np.shape(station_loc_idx)[1], 2))
        print "After load data"
        for i in range(0, np.shape(station_loc_idx)[1]):
            loc_cur = station_loc[station_loc_idx[0][i],:]
            # a = min(abs(loc_cur[0] - xi))
            # b = min(abs(loc_cur[1] - yi))
            x_cur = np.argmin(abs(loc_cur[0] - xi))
            y_cur = np.argmin(abs(loc_cur[1] - yi))
            self.idx_station_v[i]= x_cur * self.n_lat + (y_cur + 1)
            #idx_sta[y_cur, x_cur] = 1
            #sta_loc[i,:] = [y_cur, x_cur]

    def data_seperation(self, data_in, i_t):
        """
        seperate data into two parts: ground truth and training
        return: data_gt: data for ground truth
                data_pre: data for prediction model
                data_upd: data for update model
                data_ver: data for online verification
                smp_cnt_gt: sample count for ground truth
                smp_cnt_pre: sample count for prediction model
                smp_cnt_upd: sample count for update model
                smp_cnt_ver: sample count for online verification
        parameters:  data_in: input data
                     smp_cnt: sample count in each grid
                     i_t: time slot
                     alg_sep: selection method 1-randomly pickup area from all records
                     pct_gt: percent of ground truth
        """

        # selection ground
        print("data_in")
        print(data_in)
        row, col = np.nonzero(self.data.smp_cnt[i_t])
        n_val = len(row)

        n_gt = int(round(self.pct_gt * n_val))
        r_all = np.random.permutation(n_val)
        n_pre = int(round((n_val - n_gt) * self.pct_mat[0]))
        n_upd = int(round((n_val - n_gt) * self.pct_mat[1]))
        n_ver = n_val - n_gt - n_pre - n_upd
        print("data_seperation")
        print(self.flag_empty)

        if not self.flag_empty:
            # choose different area, each area get average value
            if self.alg_sep == 1:
                r_gt = np.sort(r_all[0: n_gt])
                r_pre = np.sort(r_all[n_gt: n_gt + n_pre + 1])
                r_upd = np.sort(r_all[n_gt + n_pre: n_gt + n_pre + n_upd + 1])
                r_ver = np.sort(r_all[n_gt + n_pre + n_upd: end + 1])

                for i_gt in range(len(r_gt)):
                    id_cur = r_gt[i_gt]
                    r = row[id_cur]
                    c = col[id_cur]
                    self.data.data_gt[i_t][r][c] = np.mean(data_in[r][c])
                    self.data.smp_cnt_gt[i_t][r][c] = self.data.smp_cnt[i_t][r][c]

                for i_pre in range(n_pre):
                    id_cur = r_pre[i_pre]
                    r = row[id_cur]
                    c = col[id_cur]
                    self.data.data_pre[i_t][r][c] = np.mean(data_in[r][c])
                    self.data.smp_cnt_pre[i_t][r][c] = self.data.smp_cnt[i_t][r][c]

                for i_upd in range(n_upd):
                    id_cur = r_upd[i_upd]
                    r = row[id_cur]
                    c = col[id_cur]
                    self.data.data_upd[i_t][r][c] = np.mean(data_in[r][c])
                    self.data.smp_cnt_upd[i_t][r][c] = self.data.smp_cnt[i_t][r][c]

                for i_ver in range(n_ver):
                    id_cur = r_ver[i_ver]
                    r = row[id_cur]
                    c = col[id_cur]
                    self.data.data_ver[i_t][r][c] = np.mean(data_in[r][c])
                    self.data.smp_cnt_ver[i_t][r][c] = self.data.smp_cnt[i_t][r][c]

            elif self.alg_sep == 2:
                # choose the outside as the ground truth
                c_point = [self.n_lat/2 - 1, self.n_lon/2 - 1] # python is 0-based, for caculate the distance, minus 1 first
                dis_mat = cdist(np.c_[row,col], [c_point])
                if i_t % 2 == 1:
                    indices = np.argsort(dis_mat, axis=0)[::-1] # [::-1] is for descending order
                else:
                    indices = np.argsort(dis_mat, axis=0)
                    
                for i_gt in range(n_gt):
                    r = row[indices[i_gt][0]]
                    c = col[indices[i_gt][0]]
                    self.data.data_gt[i_t][r][c] = np.mean(data_in[r][c])  # not sure about the origin matlab code
                    self.data.smp_cnt_gt[i_t][r][c] = self.data.smp_cnt[i_t][r][c]

                r_tr = np.random.permutation(n_val - n_gt) + n_gt # not sure about the origin matlab code

                for i_pre in range(n_pre):
                    id_cur = r_tr[i_pre]
                    r = row[indices[id_cur][0]]
                    c = col[indices[id_cur][0]]
                    self.data.data_pre[i_t][r][c] = np.mean(data_in[r][c])
                    self.data.smp_cnt_pre[i_t][r][c] = self.data.smp_cnt[i_t][r][c]

                for i_upd in range(n_pre):
                    id_cur = r_tr[i_upd + n_pre]
                    r = row[indices[id_cur][0]]
                    c = col[indices[id_cur][0]]
                    self.data.data_upd[i_t][r][c] = np.mean(data_in[r][c])
                    self.data.smp_cnt_upd[i_t][r][c] = self.data.smp_cnt[i_t][r][c]

                for i_ver in range(n_pre):
                    id_cur = r_tr[i_ver + n_pre + n_upd]
                    r = row[indices[id_cur][0]]
                    c = col[indices[id_cur][0]]
                    self.data.data_ver[i_t][r][c] = np.mean(data_in[r][c])
                    self.data.smp_cnt_ver[i_t][r][c] = self.data.smp_cnt[i_t][r][c]


    def baseML(self, data_cur, i_t):
        """
        system Running in time series
        data_cur: x * 7, numpy array
        parameters: data_rcd: data from records
                    alg_grid: method to combine data in each grid
                    n_lat, n_lon: latitude and longitude grids number
                    i_gas: gas index, co 3, co2 4, o3 5, pm25 6
        """
        print "data_cur"
        print data_cur
        # put record data into each grid
        data_grid_cur, self.data.smp_cnt[i_t] = rcd2grid(data_cur, self.n_lat, self.n_lon, self.flag_empty)


        print("rcd2grid finished")
        # seperate data into 4 parts
        self.data_seperation(data_grid_cur, i_t)
        print("data_seperation finished")

        # get the round truth when the data is flag_empty
        if np.count_nonzero(self.data.smp_cnt_gt[i_t]) < 3:
            self.data.data_gt[i_t] = self.data.data_gt[i_t - 1]
            self.data.smp_cnt_gt[i_t] = self.data.smp_cnt_gt[i_t - 1]

        # get result for data driven method
        data_tr = [sum(x) for x in zip(self.data.data_pre[i_t], self.data.data_upd[i_t], self.data.data_ver[i_t])] # for numpy array: data.data_pre[i_t] + data.data_udp[i_t] + data.data_ver[i_t]
        print "data_tr"
        print data_tr
        print("get get_dd_result: " + str(i_t))
        self.data.x_est_dd[i_t] = get_dd_result(data_tr, self.n_lat, self.n_lon)
        print(self.data.x_est_dd[i_t])

        print(self.flag_empty)
        if not self.flag_empty:
            if (np.count_nonzero(self.data.x_est_dd[i_t]) / (self.n_lat * self.n_lon)) <= 0.2:
                idx = self.data.x_est_dd[i_t] == 0
                print("idx")
                print(idx)
                if i_t > 0:
                    self.data.x_est_dd[i_t][idx] = self.data.x_est_dd[i_t - 1][idx]

        if i_t < self.t_len_his:
            self.data.x_est_ann[i_t] = self.data.x_est_dd[i_t]
            self.data.x_est_gp[i_t] = self.data.x_est_dd[i_t]
        else:
            sum_dd = np.zeros((self.n_lat, self.n_lon))
            # Creating a fitting network

            inputs = np.empty((0, 3), int)
            targets = np.empty((0, 1))
            for t_cur in range (i_t - self.t_len_his, i_t):
                print(t_cur)
                sum_dd = sum_dd + self.data.x_est_dd[t_cur]
                smp_cnt_gt_cur = self.data.smp_cnt_gt[t_cur]
                print(type(smp_cnt_gt_cur))
                idx = np.nonzero(smp_cnt_gt_cur > 0)
                if idx[0].size: #row index
                    row, col = idx
                    inputs_cur = np.c_[t_cur*np.ones(len(row)), row, col]
                    inputs = np.r_[inputs, inputs_cur]
                    targets = np.c_[targets, self.data.data_gt[t_cur][idx]]

            # get interpolation results
            self.data.x_est_dd[i_t] = sum_dd/self.t_len_his

            x_mat = np.tile(np.arange(0, self.n_lat), self.n_lon)
            y_mat = np.repeate(np.arange(0, self.n_lon), self.n_lat)
            test_input = np.c_[i_t * np.ones(self.n_lon * self.n_lat), x_mat, y_mat]

            # train the neural networking
            ann = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 1), random_state=1)
            ann.fit(inputs, targets.flatten())
            self.data.x_est_ann[i_t] = np.reshape(ann.predict(test_input), (self.n_lat, self.n_lon))

            # train the Gaussian gaussian_process
            gp = GaussianProcessRegressor()
            gp.fit(inputs, targets.flatten())
            self.data.x_est_gp[i_t] = np.reshape(gp.predict(test_input), (self.n_lat, self.n_lon))


        if self.flag_range_lim:
            limit_range(self.data.x_est_dd[i_t], 0, self.pre_max_val)
            limit_range(self.data.x_est_ann[i_t], 0, self.pre_max_val)
            limit_range(self.data.x_est_gp[i_t], 0, self.pre_max_val)


    def run_iter(self, i_t):
        t_low = self.date_min_num + i_t * self.res_t / 24
        t_high = self.date_min_num + (i_t + 1) * self.res_t / 24
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

        self.baseML(data_cur, i_t)

        # evaluation
        self.data.eva_re_err_dd[i_t] = get_relative_error(self.data.x_est_dd[i_t], self.data.data_gt[i_t], self.data.smp_cnt_gt[i_t], True, self.n_lat, self.n_lon)
        print ("compute error: " + str(i_t))
        print (self.data.eva_re_err_dd[i_t])
        self.data.eva_re_err_ann[i_t] = get_relative_error(self.data.x_est_ann[i_t], self.data.data_gt[i_t], self.data.smp_cnt_gt[i_t], True, self.n_lat, self.n_lon)
        self.data.eva_re_err_gp[i_t] = get_relative_error(self.data.x_est_gp[i_t], self.data.data_gt[i_t], self.data.smp_cnt_gt[i_t], True, self.n_lat, self.n_lon)

        # use station data to evaluation
        # if data_station_cur.size
        #     ds_cur = data_station_cur[1,:]
        #     ds_cur[ds_cur = 0] = 1e-4
        #     self.data.eva_all(i_t,:,:) = abs([self.data.x_est_dd[i_t](self.idx_station_v), self.data.x_est_ann[i_t](self.idx_station_v)\

        #         - repmat(np.transpose(ds_cur), 1, 3))/repmat(np.transpose(ds_cur), 1, 3)
        #     self.data.data_station_all[i_t] = ds_cur

    def data_pre_sep(self):
        with open('/Users/melody/Downloads/distribution_res_newArea.csv') as csvfile:
            readCSV = csv.reader(csvfile)
            for row in readCSV:
                time_f, lon_f, lat_f, gas_f = row
                time = int(float(time_f))
                lat = int(float(lat_f))
                lon = int(float(lon_f))
                gas = float(gas_f)
                if time not in self.data.new_data_gt:
                    self.data.new_data_gt[time] = np.zeros((self.n_lat, self.n_lon))
                    self.data.smp_cnt[time] = np.zeros((self.n_lat, self.n_lon))
                print(time)
                self.data.new_data_gt[time][lat][lon] = gas
                if gas > -1:
                    self.data.smp_cnt[time][lat][lon] = 1

        with open(r"data_bl.obj", "wb") as output:
            pickle.dump(self.data, output)

    def get_all_result(self, i_t):
        self.data_seperation(self.data.new_data_gt[i_t], i_t)
        print("data_seperation finished")

        # get the round truth when the data is flag_empty

        # get result for data driven method
        data_tr = [sum(x) for x in zip(self.data.data_pre[i_t], self.data.data_upd[i_t], self.data.data_ver[i_t])] # for numpy array: data.data_pre[i_t] + data.data_udp[i_t] + data.data_ver[i_t]
        print "data_tr"
        print data_tr
        print("get get_dd_result: " + str(i_t))
        self.data.x_est_dd[i_t] = get_dd_result(data_tr, self.n_lat, self.n_lon)

        print(self.flag_empty)
        if not self.flag_empty:
            if (np.count_nonzero(self.data.x_est_dd[i_t]) / (self.n_lat * self.n_lon)) <= 0.2:
                idx = self.data.x_est_dd[i_t] == 0
                print("idx")
                print(idx)
                if i_t > 0:
                    self.data.x_est_dd[i_t][idx] = self.data.x_est_dd[i_t - 1][idx]

        if i_t < self.t_len_his:
            self.data.x_est_ann[i_t] = self.data.x_est_dd[i_t]
            self.data.x_est_gp[i_t] = self.data.x_est_dd[i_t]
        else:
            sum_dd = np.zeros((self.n_lat, self.n_lon))
            # Creating a fitting network

            inputs = np.empty((0, 3), int)
            targets = np.empty((0, 1))
            for t_cur in range(i_t - self.t_len_his, i_t):
                sum_dd = sum_dd + self.data.x_est_dd[t_cur]
                smp_cnt_gt_cur = self.data.smp_cnt_gt[t_cur]
                idx = np.nonzero(smp_cnt_gt_cur > 0)
                if idx[0].size:  # row index
                    row, col = idx
                    inputs_cur = np.c_[t_cur * np.ones(len(row)), row, col]
                    inputs = np.r_[inputs, inputs_cur]
                    targets = np.c_[targets, self.data.data_gt[t_cur][idx]]

            # get interpolation results
            self.data.x_est_dd[i_t] = sum_dd / self.t_len_his

            x_mat = np.tile(np.arange(0, self.n_lat), self.n_lon)
            y_mat = np.repeate(np.arange(0, self.n_lon), self.n_lat)
            test_input = np.c_[i_t * np.ones(self.n_lon * self.n_lat), x_mat, y_mat]

            # train the neural networking
            ann = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 1), random_state=1)
            ann.fit(inputs, targets.flatten())
            self.data.x_est_ann[i_t] = np.reshape(ann.predict(test_input), (self.n_lat, self.n_lon))

            # train the Gaussian gaussian_process
            gp = GaussianProcessRegressor()
            gp.fit(inputs, targets.flatten())
            self.data.x_est_gp[i_t] = np.reshape(gp.predict(test_input), (self.n_lat, self.n_lon))

        if self.flag_range_lim:
            limit_range(self.data.x_est_dd[i_t], 0, self.pre_max_val)
            limit_range(self.data.x_est_ann[i_t], 0, self.pre_max_val)
            limit_range(self.data.x_est_gp[i_t], 0, self.pre_max_val)


    def run_iter_new(self, i_t):
        data_cur = self.data.new_data_gt[i_t]
        idx = np.nonzero(data_cur > 0)
        if idx[0].size:
            self.flag_empty = False
        else:
            self.flag_empty = True
        self.get_all_result(i_t)
        self.data.eva_re_err_dd[i_t] = get_relative_error(self.data.x_est_dd[i_t], self.data.new_data_gt[i_t], self.data.smp_cnt_gt[i_t], True, self.n_lat, self.n_lon)
        print ("compute error: " + str(i_t))
        print (self.data.eva_re_err_dd[i_t])
        self.data.eva_re_err_ann[i_t] = get_relative_error(self.data.x_est_ann[i_t], self.data.data_gt[i_t], self.data.smp_cnt_gt[i_t], True, self.n_lat, self.n_lon)
        self.data.eva_re_err_gp[i_t] = get_relative_error(self.data.x_est_gp[i_t], self.data.data_gt[i_t], self.data.smp_cnt_gt[i_t], True, self.n_lat, self.n_lon)


