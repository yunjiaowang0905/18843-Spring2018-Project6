#!/usr/bin/env python

"""An example script to run the system.

Usage:
    run_sys.py [-h] [-1] [-2] [-3]

 Options:
    -h, --help       Show the help
    -1, --step1      Execute step1 (baseline ML algorithms)
    -2, --step2      Execute step2 (flag learning)
    -3, --step3      Execute step3 (particle filter)

"""

__author__ = 'Nanshu Wang'

import os
import sys
import yaml
import docopt
import scipy.io
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), "util"))
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from model import particle_filter, neural_network, gaussian_process
from util.geo import deg2km
from util.data_helper import data_schema

ROOT_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
CONF_DIR = ROOT_DIR + "/conf"
DATA_DIR = ROOT_DIR + "/data"

class Scheduler():
    def __init__(self, conf):
        self.car_no = conf['car_no']
        self.date_min = datetime.strptime(conf['date_min'], "%Y-%m-%d %H:%M:%S")
        self.date_max = datetime.strptime(conf['date_max'], "%Y-%m-%d %H:%M:%S")
        self.date_min_num = date_min.toordinal() + 366
        self.date_max_num = date_max.toordinal() + 366
        self.res_t = conf['rest_t']
        self.n_time = (self.date_max-self.date_min).seconds / 3600 / self.res_t

        self.lon_min = conf['lon_min']
        self.lon_max = conf['lon_max']
        self.lat_min = conf['lat_min']
        self.lat_max = conf['lat_max']

        self.res_s = conf['res_s']
        self.n_lon = np.ceil(deg2km(self.lon_max-self.lon_min) / self.res_s * 1000)
        self.n_lat = np.ceil(deg2km(self.lat_max-self.lat_min) / self.res_s * 1000)
        # self.l = self.res_s
        self.pct_gt = conf['pct_gt']
        self.pct_mat = [conf['pct_pred'], conf['pct_corr'], conf['pct_veri']]
        self.gas_max_val = conf['gas_max_val']
        self.flag_range_lim = conf['flag_range_lim']
        self.w_mat = np.array([-0.3,-0.3,-0.3,0.8,0.5])

        self.data = data_schema(self.n_time)

        self.eva_all = np.zeros((self.n_time, 5, 3))
        self.eva_all_pf = np.zeros((self.n_time, 5, 3))

    def pf_run(self, conf):
        self.data.load(DATA_DIR)
        i_t = 0
        pf = particle_filter(conf, self.res_t, self.res_s, self.n_lat, self.n_lon, self.data)

        while i_t < self.n_time:
            t_low = self.date_min + timedelta(hours=i_t * self.res_t)
            # t_high = self.date_min + timedelta(hours=(i_t+1) * self.res_t / 24)
            print(t_low)
            pf.run_iter(i_t)
            i_t += 1

    def baseML_run(self):
        # select the data for pollution map reconstruction and calibrate the data
        data_slt, xi, yi = data_pre_slt(DATA_DIR+ "/data/201701/raw/slt_raw_area_time_early.mat"）

        # get the data from monitoring station
        station_info = scipy.io.loadmat(DATA_DIR + "/data/station_info.mat")
        station_data = scipy.io.loadmat(DATA_DIR + "/data/station_data.mat")
        station_info_lon = np.asarray([v[1] for v in station_info['station_info']]).flatten()
        station_info_lat = np.asarray([v[2] for v in station_info['station_info']]).flatten()
        station_loc = np.c_[station_info_lon, station_info_lat]
        station_loc_idx = np.nonzero((station_loc[:,0] >= self.lon_min) & (station_loc[:,0] <= self.lon_max) \
            & (station_loc[:,1] >= self.lat_min) & (station_loc[:,1] <= self.lat_max))
        station_data_loc = station_data['station_data'][:, np.r_[0, station_loc_idx[0] + 2]]
        station_data_idx = np.nonzero((station_data_loc[:,0] >= date_min_num) & (station_data_loc[:,0] <= date_max_num))
        station_data_slt = station_data_loc[station_data_idx[0],:]
        pre_max_val = np.max(station_data_slt[:,1:len(station_data_slt[0])])
        idx_station_v = np.zeros(np.shape(station_loc_idx))
        idx_sta = np.zeros(self.n_lat, self.n_lon) # n_lon=64, n_lat=16
        sta_loc = np.zeros((np.shape(station_loc_idx)[1], 2))

        for i in range(0, np.shape(station_loc_idx)[1]):
            loc_cur = station_loc[station_loc_idx[0][i],:]
            a = min(abs(loc_cur[0] - xi))
            b = min(abs(loc_cur[1] - xi))
            x_cur = np.argmin(abs(loc_cur[0] - yi))
            y_cur = np.argmin(abs(loc_cur[1] - yi))
            idx_station_v[i, 0]= x_cur * n_lat + (y_cur + 1)
            idx_sta[y_cur, x_cur] = 1
            sta_loc[i,:] = [y_cur, x_cur]


if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    execute_steps = [False] \
        + [args["--step{}".format(step_index)] for step_index in range(1, 4)]

    if not any(execute_steps):
        print 'Please specify which step to execute. Try "run_sys.py -h" for help. '

    with open(CONF_DIR + "/default.yml", "r") as f:
        conf = yaml.load(f)

    sess = Scheduler(conf['sys'])

    if execute_steps[1]:
        print '### 1. run baseline ML algorithms ###'
        sess.baseML_run()

    if execute_steps[2]:
        print '### 2. run flag learn ###'
        pass

    if execute_steps[3]:
        print '### 3. run particle filter algorithm ###'
        sess.pf_run(conf['particle_filter'])
