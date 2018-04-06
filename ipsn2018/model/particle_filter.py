from __future__ import absolute_import, division, print_function
__author__ = 'Nanshu Wang'

import numpy as np
import sys, os
from random import random
from scipy.linalg import expm, solve
from scipy.sparse import bsr_matrix
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from scipy.ndimage import iterate_structure
from util.tomPhy import predictTimeSlice
from util.correction_strategy import predictFlag
sys.path.insert(0, os.path.abspath(".."))

from util import getA_c, get_par2, get_next_state, limit_range

class particle_filter(object):
    """Particle Filter
    This class offers the training and prediction of Particle Filter
    A Particle Filter is a sequential Monte Carlo method for on-line state
    tracking, which works within a Bayesian framework and under Markov
    assumption. For air pollution problem, the two stages of Particle
    Filter is: 1) state evolution estimation by physical model;
    2) estimation correction by adaptively correct physical model with data
    model
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
        self.pre_max_val = 2541
        self.flag_cal_feature = 1

    def generate_observation(self, i_t):
        self.data.data_upd_interp[i_t] = limit_range(self.data.x_est_gp[i_t], 0, self.pre_max_val)

    def predict(self, i_t):
        """
        physical guided model for state evolution estimation
        x_P_update: dimension (N_pf, n_lat, n_lon) particle filter particles,
        x_P_update[i, j, k] denote the ith particle value for subarea (j,k)
        C[k + 1] = AC[k] + BU[k]
        """
        # est_history = self.data.x_est_adp[i_t-self.t_len_his:i_t]
        # u_mat_tmp = get_par2(self.coe_matrix, self.B, est_history)
        # self.data.u_mat_adp[i_t] = u_mat_tmp
        self.data.x_P_update = np.zeros((self.N_pf, self.n_lat, self.n_lon))
        gp = self.data.x_est_gp
        pred = predictTimeSlice(gp[i_t-2], gp[i_t-1], gp[i_t], self.data.data_train[i_t])
        # update each particle filter with previous value
        for i_pf in range(self.N_pf):
            # x_est_pre = self.data.x_P_adp[i_t-1][i_pf,:,:]
            # self.data.x_P_update[i_pf, :, :] = get_next_state(self.A, self.B, x_est_pre, u_mat_tmp,self.coe_matrix) \
            #                                    + np.sqrt(self.x_N_pf) * np.random.randn(self.n_lat, self.n_lon)

            self.data.x_P_update[i_pf,:,:] = pred + np.sqrt(self.x_N_pf) * np.random.randn(self.n_lat, self.n_lon)

        if self.flag_range_lim == 1:    # range control
            self.data.x_P_update = limit_range(self.data.x_P_update, 0, self.pre_max_val)

    def estimation_entropy(self, p_mat, x):
        n_bin = np.size(p_mat)
        # c_bin = np.linspace(0, self.pre_max_val, n_bin)
        n_p = np.zeros((n_bin, 1)) + 1e-15 # prevent zero possibilities
        for i in range(n_bin):
            idx = max(0, min(self.N_pf-1, np.int(np.floor(x[i] / (self.pre_max_val / n_bin)))))
            n_p[idx] += p_mat[i]

        E = -sum(n_p * np.log2(n_p))
        return E

    def nearest_neighbour_distances(self, i_y, i_x, col, row):

        m_d = 8*4 # TODO remove magic number
        dis = m_d
        n_upd = col.shape[0]
        # feature_mat[i][1] is the distance sum of 8 element
        if n_upd != 0:
            dis_mat = cdist(zip(col, row),[[i_y, i_x]])
            dis_mat = np.sort(dis_mat)

        if n_upd >= 8:
            dis = sum(dis_mat[0:8])
        elif n_upd > 0:
            dis = sum(dis_mat) / n_upd * 8

        dis = min(dis, m_d)

        return dis

    def calculate_feature(self, i_t):
        """
        :return: distance: dimension = (n_lat, n_lon)
        For subarea(i,j), distance[i, j] is the sum of M=8 nearest neighbours' distances.
        The large distance, the better quality of generated measurement of subarea(i,j)

        entropy[i, j] is the entropy of physics guided state evolution estimate distribution
        the large entropy, the more consistent of the predictions in all the particles, the better
        quality of generated measurement
        """
        distance = np.zeros((self.n_lat, self.n_lon))
        entropy = np.zeros((self.n_lat, self.n_lon))
        if self.flag_cal_feature:
            col, row = np.nonzero(self.data.smp_cnt_upd[i_t] > 0)
            for i_y in range(self.n_lat):
                for i_x in range(self.n_lon):
                    x_cur = np.reshape(self.data.x_P_adp[i_t-1][:, i_y, i_x], (self.N_pf,1))
                    Pw_cur = np.reshape(self.data.P_w_adp[i_t-1][:, i_y, i_x], (self.N_pf,1))
                    distance[i_y, i_x] = self.nearest_neighbour_distances(i_y, i_x, col, row)
                    entropy[i_y, i_x] = self.estimation_entropy(Pw_cur, x_cur)

            # normalization
            distance = distance
            entropy = entropy

        return distance, entropy

    def get_upd_flag(self, i_t, alg_upd=3):
        """
        :param i_t:
        :param distance: sum of M nearest neighbours' distances, D
        :param entropy: entropy of physics guided state evolution estimate distribution, E
        :return: pf_upd_flag: dimension (n_lat, n_lon)
        pf_upd_flag[i,j] = 1 denotes skip estimate correction for the subarea(i,j), i,e the
        equation: E - alpha * D > 0 sustain.
        """
        pf_upd_flag = np.zeros((self.n_lat, self.n_lon))
        if self.flag_empty(i_t):
            return pf_upd_flag

        # need to be modified
        # all areas update
        if alg_upd == 1:
            pf_upd_flag = np.ones((self.n_lat, self.n_lon))
        # update the collected data areas
        elif alg_upd == 2:
            idx = self.data.smp_cnt_upd[i_t]>0
            pf_upd_flag[idx] = 1
        # update collected data areas + good compensated data areas + dilation --- need to modify
        elif alg_upd == 3:
            idx = np.nonzero(self.data.smp_cnt_upd[i_t] > 0)
            pf_upd_flag[idx] = 1

            struct = generate_binary_structure(2, 1) # se = strel('square',3);
            se = iterate_structure(struct, 3).astype(int)
            pf_upd_flag = binary_dilation(pf_upd_flag, structure=struct).astype(float)
        # for adaptive scheme
        elif self.alg_upd == 4:
            distance, entropy = self.calculate_feature(i_t)
            feature_tmp = np.reshape(distance - entropy, (self.n_lat, self.n_lon))
            idx = feature_tmp <= 0
            pf_upd_flag[idx] = 1
            idx1 = self.data.smp_cnt_upd[i_t]>0
            pf_upd_flag[idx1] = 1
        elif self.alg_upd == 5:
            idx = np.nonzero(self.data.smp_cnt_upd[i_t] > 0)
            pf_upd_flag[idx] = 1
            struct = generate_binary_structure(2, 1) # se = strel('square',3);
            se = iterate_structure(struct, 3).astype(int)
            pf_upd_flag = binary_dilation(pf_upd_flag, structure=se).astype(float)
            pf_upd_flag = predictFlag(self.data.pf_upd_flag_adp[i_t-self.t_len_his:i_t], pf_upd_flag)
        return pf_upd_flag

    def update(self, i_t):
        """
        update weight of particles P_w_adp based on pf_upd_flag_adp
        """
        matrix_shape = (self.N_pf, 1, 1)
        pf_upd_flag_new = np.tile(self.data.pf_upd_flag_adp[i_t], matrix_shape)
        idx = np.nonzero(pf_upd_flag_new == 1)
        z = np.tile(self.data.data_upd_interp[i_t], matrix_shape)
        P_w_tmp = np.zeros((self.N_pf, self.n_lat, self.n_lon))
        diff = z[idx] - self.data.x_P_update[idx]
        coef = 1 / np.sqrt(2 * np.pi * self.x_R_pf)
        P_w_tmp[idx] = coef * np.exp(-diff ** 2 / (2 * self.x_R_pf))
        P_w_tmp = P_w_tmp + 1e-15
        P_w_tmp = P_w_tmp / np.tile(sum(P_w_tmp, 0), matrix_shape)
        self.data.P_w_adp[i_t] = P_w_tmp

    def resample(self, i_t):
        # TODO why ??
        """
        for subarea(i,j), if skip correction, x_P_adp = x_P_adp_update
        otherwise, x_P_adp = randomly choose x_P_update based on weight of particles.
        estimation = weighted average of all particles
        """
        self.data.x_P_adp[i_t] = self.data.x_P_update
        col, row = np.nonzero(self.data.pf_upd_flag_adp[i_t] > 0.1)
        print(len(col))
        for i_x, i_y in zip(row, col):
            # P_w_cur = self.data.P_w_adp[i_t][:, i_y, i_x].reshape((self.N_pf, 1, 1)).copy()
            # x_P_update_cur = self.data.x_P_update[:, i_y, i_x]
            for i_pf in range(self.N_pf):
            #     rand = random()
            #     idx = np.nonzero(rand <= np.cumsum(P_w_cur))
                # self.data.x_P_adp[i_t][i_pf, i_y, i_x] = x_P_update_cur[idx][0]
                self.data.x_P_adp[i_t][i_pf, i_y, i_x] = self.data.x_est_gp[i_t][i_y][i_x] + np.sqrt(self.x_N_pf) * np.random.randn()

        if self.flag_range_lim==1:
            # this is only for CO value range limitation
            self.data.x_P_adp[i_t] = limit_range(self.data.x_P_adp[i_t], 0, self.pre_max_val)

        # weighted average
        self.data.x_est_adp[i_t] = np.mean(self.data.x_P_adp[i_t], 0)

    def initialize_stage(self, i_t):
        X0 = self.data.data_upd_interp[i_t]
        # x_P_adp has dimension of (N_pf, n_lat, n_lon) = (100,16,64) this is different from original matlab (16,64,100)
        self.data.x_P_adp[i_t] = np.tile(X0, (self.N_pf, 1, 1)) + \
                                 np.sqrt(self.V_pf) * np.random.randn(self.N_pf, self.n_lat, self.n_lon)
        self.data.x_est_adp[i_t] = X0
        self.data.P_w_adp[i_t] = np.ones((self.N_pf, self.n_lat, self.n_lon)) / self.N_pf
        self.data.pf_upd_flag_adp[i_t] = self.get_upd_flag(i_t, 3)
        print("Initialization iteration")

    def flag_empty(self, i_t):
        return np.count_nonzero(self.data.smp_cnt_upd[i_t]) == 0

    def run_iter(self, i_t):
        # use x_est_gp as input
        self.generate_observation(i_t)

        self.flag_cal_feature = 1

        if i_t < self.t_len_his:
            self.initialize_stage(i_t)
        else:
            self.predict(i_t)
            # self.data.x_P_update = np.transpose(self.data.x_P_adp[i_t], (2,0,1))

            self.data.pf_upd_flag_adp[i_t] = self.get_upd_flag(i_t, self.alg_upd)
            self.update(i_t)
            self.resample(i_t)

        self.flag_range_lim = 1
