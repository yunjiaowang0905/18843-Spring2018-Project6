from __future__ import absolute_import, division, print_function

import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(".."))

from util import getA_c

class particle_filter(object):
    """Particle Filter
    This class offers the training and prediction of Particle Filter
    Parameters
    """
    def __init__(self, conf, res_t, res_s, n_lat, n_lon, data):
        """
        :param conf: configure for particle filter, please refer conf/default.yml
        :param res_t: time resolution in seconds
        :param res_s: space resolution in meters
        :param n_lat: latitude grids number
        :param n_lon: longitude grids number
        :param data: data_schema class, please refer util/data_helper.py
        """
        self.res_t = res_t
        self.res_s = res_s
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.data = data

        self.x_N_pf = conf['x_N_pf'] * conf['x_N_pf']
        self.x_R_pf = conf['x_R_pf'] * conf['x_R_pf']
        self.N_pf = conf['N_pf']
        self.V_pf = self.x_R_pf

        self.alg_upd = conf['alg_upd']
        self.alg_sep = conf['alg_sep']

        self.t_len_his = conf['t_len_his']
        self.ver_re_err_th = conf['ver_re_err_th']
        self.ver_var_th = conf['ver_var_th']

        self.feature_mat_all = []
        self.feature_sta_mat_all = []

        self.K = conf['K']
        self.vx = conf['vx']
        self.vy = conf['vy']
        self.dt = res_t * 60 * 60

        self.V_x = np.zeros((n_lat, n_lon)) + self.vx
        self.V_y = np.zeros((n_lat, n_lon)) + self.vy
        self.A = getA_c(self.K, res_s, self.V_x, self.V_y)

        self.flag_range_lim = 1
        self.pre_max_val = 10

    def generate_observation(self, i_t):
        self.data.data_upd_interp[i_t] = self.data.x_est_gp[i_t]
        self.data.data_upd_interp[i_t][self.data.data_upd_interp[i_t]<0] = 0
        self.data.data_upd_interp[i_t][self.data.data_upd_interp[i_t]>self.pre_max_val] = self.pre_max_val

    def predict(self, i_t):
        pass

    def calculate_feature(self, i_t):
        pass

    def get_update_flag(self, i_t):
        pass

    def update(self, i_t):
        pass

    def resample(self, i_t):
        pass

    def initialize_stage(self, i_t):
        X0 = self.data.data_upd_interp[i_t]
        self.data.x_P_adp[i_t] = np.tile(X0, (100, 1, 1)) + \
                                 np.sqrt(self.V_pf) * np.random.randn(self.n_lat, self.n_lon, self.N_pf)
        self.data.x_est_adp[i_t] = X0
        self.data.P_w_adp[i_t] = np.ones((100, self.n_lat, self.n_lon)) / 100
        self.data.pf_upd_flag_adp[i_t] = np.zeros((self.n_lat, self.n_lon))
        print("Initialization iteration")

    def flag_empty(self, i_t):
        return np.count_nonzero(self.data.smp_cnt_upd[i_t]) == 0

    def run_iter(self, i_t):
        self.generate_observation(i_t)
        self.alg_upd = 4
        self.flag_range_lim = 1
        self.flag_cal_feature = 1

        if i_t < self.t_len_his + 1:
            self.initialize_stage(i_t)
        else:
            self.predict(i_t)
            self.calculate_feature(i_t)
            self.get_update_flag(i_t)
            self.update(i_t)
            self.resample(i_t)

        self.flag_range_lim = 0