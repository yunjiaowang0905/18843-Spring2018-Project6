#!/usr/bin/env python

"""An example script to run the system.

Usage:
    run_sys.py [-h] [-1] [-2] [-3] [-4]

 Options:
    -h, --help       Show the help
    -1, --step1      Execute step1 (baseline ML algorithms)
    -2, --step2      Execute step2 (flag learning)
    -3, --step3      Execute step3 (particle filter)
    -4, --step4      Execute step4 (evaluate result)

"""

__author__ = 'Nanshu Wang'

import os
import sys
import yaml
import docopt
import pickle
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), "util"))
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
# from model import particle_filter, baselines
from model import particle_filter
from util.geo import deg2km
from util.data_helper import data_schema
from util.visualize import show_result

ROOT_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
CONF_DIR = ROOT_DIR + "/conf"
DATA_DIR = ROOT_DIR + "/data"

class Scheduler():
    def __init__(self, conf):
        self.car_no = conf['car_no']
        self.date_min = datetime.strptime(conf['date_min'], "%Y-%m-%d %H:%M:%S")
        self.date_max = datetime.strptime(conf['date_max'], "%Y-%m-%d %H:%M:%S")
        self.res_t = conf['rest_t']
        delta = self.date_max-self.date_min
        # self.n_time = delta.seconds / 3600 / self.res_t + delta.days * 24 / self.res_t
        # self.n_time = 5930
        self.n_time = 200
        self.lon_min = conf['lon_min']
        self.lon_max = conf['lon_max']
        self.lat_min = conf['lat_min']
        self.lat_max = conf['lat_max']

        self.res_s = conf['res_s']
        self.n_lon = np.int(np.ceil(deg2km(self.lon_max-self.lon_min) / self.res_s * 1000))
        self.n_lat = np.int(np.ceil(deg2km(self.lat_max-self.lat_min) / self.res_s * 1000))
        # self.n_lon = 12
        # self.n_lat = 60
        # self.l = self.res_s
        self.pct_gt = conf['pct_gt']
        self.pct_mat = [conf['pct_pred'], conf['pct_corr'], conf['pct_veri']]
        self.gas_max_val = conf['gas_max_val']
        self.flag_range_lim = conf['flag_range_lim']
        self.w_mat = np.array([-0.3,-0.3,-0.3,0.8,0.5])

        self.data = data_schema(self.n_time)

        self.eva_all = np.zeros((self.n_time, 5, 3))
        self.eva_all_pf = np.zeros((self.n_time, 5, 3))
        self.flag_empty = False

        self.result_path = conf['result_path']


    def pf_run(self, conf):
        self.data.load_from_mat(DATA_DIR)
        # self.data.load_new_mat(DATA_DIR)
        i_t = 0
        pf = particle_filter(conf, self.res_t, self.res_s, self.n_lat, self.n_lon, self.data)

        while i_t < self.n_time:
            t_low = self.date_min + timedelta(hours=i_t * self.res_t)
            # t_high = self.date_min + timedelta(hours=(i_t+1) * self.res_t / 24)
            print(str(i_t) + " " + str(t_low))
            pf.run_iter(i_t)
            i_t += 1

        self.save()

    def eval_run(self):
        self.load()
        # self.data.load_from_mat(DATA_DIR)
        show_result(self.data, self.n_time)

    def baseML_run(self):
        # select the data for pollution map reconstruction and calibrate the data
        i_t = 0
        # bl = baselines(conf, self.flag_empty, self.pct_mat, self.date_min, self.date_max, self.n_lat, self.n_lon, self.data)

        while i_t < self.n_time:
            # bl.run_iter(i_t)
            i_t += 1

    def save(self):
        with open(self.result_path, "w") as f:
            pickle.dump(self.data, f)

    def load(self):
        with open(self.result_path, "r") as f:
            self.data = pickle.load(f)


if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    execute_steps = [False] \
        + [args["--step{}".format(step_index)] for step_index in range(1, 5)]

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

    if execute_steps[4]:
        print '### 4. evaluate result ###'
        sess.eval_run()
