__author__ = 'Nanshu Wang'
import numpy as np
import scipy.io

class data_schema():
    def __init__(self, n_time):
        self.data_gt = [{} for _ in range(n_time)]   # ground truth data
        self.data_pre = [{} for _ in range(n_time)]  # data for prediction
        self.data_upd = [{} for _ in range(n_time)]  # data for update
        self.data_ver = [{} for _ in range(n_time)]  # data for online verification  --- will be modified
        self.smp_cnt = [{} for _ in range(n_time)]
        self.smp_cnt_gt = [{} for _ in range(n_time)]
        self.smp_cnt_pre = [{} for _ in range(n_time)]
        self.smp_cnt_upd= [{} for _ in range(n_time)]
        self.smp_cnt_ver = [{} for _ in range(n_time)]
        self.data_upd_interp = [{} for _ in range(n_time)]   # compensated data

        # for just prediction
        self.x_est_jp = [{} for _ in range(n_time)] # estimated state value
        self.x_P_jp = [{} for _ in range(n_time)]    # particles
        self.P_w_jp = [{} for _ in range(n_time)]   # weights
        self.u_mat_jp = [{} for _ in range(n_time)] # source
        self.pf_upd_flag_jp = [{} for _ in range(n_time)]   # flag matrix for particle filter update
        self.eva_re_err_jp = [{} for _ in range(n_time)]

        # for always update all
        self.x_est_aua = [{} for _ in range(n_time)] # estimated state value
        self.x_P_aua = [{} for _ in range(n_time)]    # particles
        self.P_w_aua = [{} for _ in range(n_time)]   # weights
        self.u_mat_aua = [{} for _ in range(n_time)] # source
        self.pf_upd_flag_aua = [{} for _ in range(n_time)]   # flag matrix for particle filter update
        self.eva_re_err_aua = [{} for _ in range(n_time)]

        # for always update data area
        self.x_est_aud = [{} for _ in range(n_time)] # estimated state value
        self.x_P_aud = [{} for _ in range(n_time)]    # particles
        self.P_w_aud = [{} for _ in range(n_time)]   # weights
        self.u_mat_aud = [{} for _ in range(n_time)] # source
        self.pf_upd_flag_aud = [{} for _ in range(n_time)]   # flag matrix for particle filter update
        self.eva_re_err_aud = [{} for _ in range(n_time)]

        # for adaptive method
        self.x_est_adp = [{} for _ in range(n_time)] # estimated state value
        self.x_P_adp = [{} for _ in range(n_time)]    # particles
        self.P_w_adp = [{} for _ in range(n_time)]   # weights
        self.u_mat_adp = [{} for _ in range(n_time)] # source
        self.pf_upd_flag_adp = [{} for _ in range(n_time)]   # flag matrix for particle filter update
        self.eva_re_err_adp = [{} for _ in range(n_time)]
        self.ver_re_err_adp = [{} for _ in range(n_time)]
        self.ver_var_adp = [{} for _ in range(n_time)]

        self.min_idx = [{} for _ in range(n_time)]

        # for data-driven method
        self.x_est_dd = [{} for _ in range(n_time)]
        self.eva_re_err_dd = [{} for _ in range(n_time)]
        self.x_est_ann = [{} for _ in range(n_time)]
        self.eva_re_err_ann = [{} for _ in range(n_time)]
        self.x_est_gp = [{} for _ in range(n_time)]
        self.eva_re_err_gp = [{} for _ in range(n_time)]

        self.data_station_all= np.zeros((n_time, 5))

    def save(self):
        pass
        # TODO

    def load(self, DATA_DIR):
        # TODO load data from specific directory
        path = DATA_DIR + "/test/gt_40_upd100_sep2_tlenhis3_rangeLim0_ann_dd_gp.mat"
        mat = scipy.io.loadmat(path)
        for key, value in mat.iteritems():
            if not key.startswith("__"):
                setattr(self, key, [v[0] for v in value])
