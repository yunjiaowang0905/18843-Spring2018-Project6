from __future__ import absolute_import, division, print_function
__author__ = 'Nanshu Wang'

import numpy as np
import sys, os
from random import random
from scipy.linalg import expm, solve
from scipy.sparse import bsr_matrix
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
sys.path.insert(0, os.path.abspath(".."))

from util import getA_c, get_par2, cvx_solve_u, get_next_state, limit_range

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
        self.gas_max_val = conf['gas_max_val']

        self.feature_mat_all = []
        self.feature_sta_mat_all = []

        self.K = conf['K']
        self.vx = conf['vx']
        self.vy = conf['vy']
        self.dt = res_t * 60 * 60

        self.V_x = np.zeros((n_lat, n_lon)) + self.vx
        self.V_y = np.zeros((n_lat, n_lon)) + self.vy
        self.A = getA_c(self.K, res_s, self.V_x, self.V_y)
        self.A = self.A - np.diag(np.sum(self.A, axis=1))
        self.A = self.A - np.diag(np.diag(self.A)) * 1e-6

        A1 = expm(self.A * self.dt)
        self.B = solve(self.A, A1 - np.eye(A1.shape[0]))
        A1[np.where(np.absolute(A1) < 1e-3 * np.amax(np.absolute(A1)))] = 0
        self.B[np.where(np.absolute(self.B) < 1e-3 * np.amax(np.absolute(self.B)))] = 0
        self.coe_matrix = bsr_matrix(A1)
        self.B = bsr_matrix(self.B)

        self.flag_range_lim = 1
        self.pre_max_val = 10
        self.flag_cal_feature = 1

    def generate_observation(self, i_t):
        self.data.data_upd_interp[i_t] = limit_range(self.data.x_est_gp[i_t], 0, self.pre_max_val)

    def predict(self, i_t):
        est_history = self.data.x_est_adp[i_t-self.t_len_his:i_t-1]
        u_mat_tmp = get_par2(self.coe_matrix, self.B, est_history)
        self.data.u_umat_adp[i_t] = u_mat_tmp
        self.data.x_P_update = np.zeros((self.n_lat, self.n_lon, self.N_pf))

        for i_pf in range(self.N_pf):
            x_est_pre = self.data.x_P_adp[i_t-1][:,:,i_pf]
            self.data.x_P_update[:, :, i_pf] = get_next_state(self.A, self.B, x_est_pre, u_mat_tmp,self.coe_matrix) \
                                               + np.sqrt(self.x_N_pf) * np.random.randn(self.n_lat, self.n_lon)

        if self.flag_range_lim == 1:    # range control
            self.data.x_P_update = limit_range(self.data.x_P_update, 0, self.gas_max_val)

    def entropy_cal(self, p_mat, x, x_min, x_max):
        n_bin = np.size(p_mat)
        c_bin = np.linspace(x_min,x_max,n_bin)
        n_p = np.zeros(n_bin,1) + 1e-15 # prevent zero possibilities
        for i in range(n_bin):
            idx = np.nonzero(c_bin < x(i))
            if len(idx): # idx not empty
                n_p[idx[-1]] = n_p[idx[-1]] + p_mat(i) # end == -1
            else:
                n_p[0] = n_p[0] + p_mat(i)

        E = -sum(n_p * np.log2(n_p))
        return E

    def get_flag_feature(self, idx, idx_upd, x_cur, Pw_cur, val_min, val_max):
        n_feature, _ = np.shape(idx) # TODO question? 1 or i_y
        feature_mat = np.zeros((n_feature,2))
        m_d = 8*4
        n_upd, _ = np.shape(idx_upd)
        for i in range(n_feature):
            i_y = idx[i][0]
            i_x = idx[i][1]
            dis_mat = cdist(idx_upd,[i_y,i_x])
            dis_mat = np.sort(dis_mat)

            if n_upd >= 8:
                feature_mat[i][0] = sum(dis_mat[0:8])
            elif n_upd > 0:
                feature_mat[i][0] = sum(dis_mat) * 8 / n_upd
            else:
                feature_mat[i][0] = m_d
            feature_mat[i][1] = self.entropy_cal(Pw_cur, x_cur, val_min, val_max)
        return feature_mat

    def calculate_feature(self, i_t):
        feature_all = np.zeros((self.n_lat, self.n_lon,2))
        if self.flag_cal_feature:
            col, row = np.nonzero(self.data.smp_cnt_upd[i_t] > 0)
            for i_y in range(self.n_lat):
                for i_x in range(self.n_lon):
                    x_cur = np.reshape(self.data.x_P_adp[i_t-1][i_y, i_x, :], (100,1))
                    Pw_cur = np.reshape(self.data.P_w_adp[i_t-1][i_y, i_x, :], (100,1))
                    feature_all[i_y, i_x, :] = self.get_flag_feature([[i_y,i_x]], [col,row], x_cur,
                                                                Pw_cur, 0, self.pre_max_val)

            # normalization
            feature_all[:, :, 0] = (feature_all[:, :, 0]-8) / 24 / 2
            feature_all[:, :, 1] = feature_all[:, :, 1] / 8

        return feature_all

    def get_upd_flag(self, i_t, feature_all):
        pf_upd_flag = np.zeros((self.n_lat, self.n_lon))
        if self.flag_empty(i_t):
            return pf_upd_flag

        # need to be modified
        # all areas update
        if self.alg_upd == 1:
            pf_upd_flag = np.ones((self.n_lat, self.n_lon))
        # update the collected data areas
        elif self.alg_upd == 2:
            if i_t >= self.t_len_his+1:
                idx = self.data.smp_cnt_upd[i_t]>0
                pf_upd_flag[idx] = 1
        # update collected data areas + good compensated data areas + dilation --- need to modify
        elif self.alg_upd == 3:
            if i_t >= self.t_len_his+1:
                idx = self.data.ver_re_err_adp[i_t-1] >= self.ver_re_err_th
                pf_upd_flag[idx] = 1

                idx = self.data.smp_cnt_upd[i_t] > 0
                pf_upd_flag[idx] = 1

                idx = self.data.ver_var_adp[i_t-1] >= self.ver_var_th
                pf_upd_flag[idx] = 1

                idx = np.nonzero(pf_upd_flag > 0)
                se = generate_binary_structure(2, 3) # se = strel('square',3);
                pf_upd_flag[idx] = binary_dilation(pf_upd_flag[idx], structure=se) # pf_upd_flag[idx] = imdilate(pf_upd_flag[idx],se);
                # use ver_re_err and ver_var at i_t - 1 to get the update flag matrix
        # for adaptive scheme
        elif self.alg_upd == 4:
            feature_tmp = np.reshape(feature_all[:,:,0] - feature_all[:,:,1], self.n_lat, self.n_lon)
            idx = feature_tmp <= 0
            pf_upd_flag[idx] = 1
            idx1 = self.data.smp_cnt_upd[i_t]>0
            pf_upd_flag[idx1] = 1

        return pf_upd_flag

    def update(self, i_t):
        pf_upd_flag_new = np.tile(self.data.pf_upd_flag_adp[i_t][0] , (1, 1, self.N_pf))
        idx = np.nonzero(pf_upd_flag_new == 1)
        z = np.tile(self.data.data_upd_interp[i_t], (1,1, self.N_pf))
        P_w_tmp = np.zeros(self.n_lat, self.n_lon, self.N_pf)
        P_w_tmp[idx] = (1/np.sqrt(2 * np.pi * self.x_R_pf)) * \
                       np.exp(-(z[idx] - self.data.x_P_update[idx]) ** 2 / (2 * self.x_R_pf))
        P_w_tmp = P_w_tmp + 1e-15
        P_w_tmp = P_w_tmp / np.tile(sum(P_w_tmp,3),[1,1,100])
        self.data.P_w_adp[i_t] = P_w_tmp

    def resample(self, i_t):
        self.data.x_P_adp[i_t, 0] = self.data.x_P_update
        row, col = np.nonzero(self.data.pf_upd_flag_apd[i_t, 0] == 1)
        # xp{i_t,1} = x_P_update;
        # [row,col] = find(pf_upd_flag_cur{i_t,1}==1);
        for id in range(row.shape[0]):
            i_x = col[id]
            i_y = row[id]
            P_w_cur = self.data.P_w_adp[i_y, i_x].reshape(1,1,self.N_pf).copy()
            x_P_update_cur = self.data.x_P_update[i_y, i_x]
            for i_pf in range(self.N_pf):
                rand = random()
                self.data.x_P_adp[i_t,0,i_y,i_x,i_pf] = x_P_update_cur[rand <= np.cumsum(P_w_cur),1]

        if self.flag_range_lim==1:
            # this is only for CO value range limitation
            self.data.x_P_adp[i_t,0] = limit_range(self.data.x_P_adp[i_t,0], 0, self.pre_max_val)

        # weighted average
        self.data.x_est_adp[i_t,0] = np.mean(self.data.x_est_adp[i_t], axis=3)

    def initialize_stage(self, i_t):
        X0 = self.data.data_upd_interp[i_t]
        # x_P_adp has dimension of (N_pf, n_lat, n_lon) = (100,16,64) this is different from original matlab (16,64,100)
        self.data.x_P_adp[i_t] = np.tile(X0, (self.N_pf, 1, 1)) + \
                                 np.sqrt(self.V_pf) * np.random.randn(self.N_pf, self.n_lat, self.n_lon)
        self.data.x_est_adp[i_t] = X0
        self.data.P_w_adp[i_t] = np.ones((self.N_pf, self.n_lat, self.n_lon)) / self.N_pf
        self.data.pf_upd_flag_adp[i_t] = np.zeros((self.n_lat, self.n_lon))
        print("Initialization iteration")

    def flag_empty(self, i_t):
        return np.count_nonzero(self.data.smp_cnt_upd[i_t]) == 0

    def run_iter(self, i_t):
        self.generate_observation(i_t)
        self.alg_upd = 4

        self.flag_cal_feature = 1

        if i_t < self.t_len_his:
            self.initialize_stage(i_t)
        else:
            self.predict(i_t)
            feature_all = self.calculate_feature(i_t)
            self.data.pf_upd_flag_adp[i_t] = self.get_upd_flag(i_t, feature_all)
            self.update(i_t)
            self.resample(i_t)

        self.flag_range_lim = 0