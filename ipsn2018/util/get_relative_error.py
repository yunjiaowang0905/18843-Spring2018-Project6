__author__ = 'Jin Gong'
import numpy as np

def get_relative_error(x_est_cur, data_gt_cur, smp_cnt_gt_cur, flag):
    """
    calculate the relative errors
    output: re_err_cur: relative error matrix
    parameters:
        x_est_cur: estimated status result matrix
        data_gt_cur: ground truth matrix
        smp_cnt_gt_cur: ground truth sample number matrix
        flag: if flag == 1, get relative error, if flag == 0, get absolute error
    """
    re_err_cur = np.zeros(n_lat, n_lon)
    idx = np.nonzero(smp_cnt_gt_cur)
    if idx[0].size:
        if flag:
            re_err_cur[idx] = abs(x_est_cur[idx] - data_gt_cur[idx]) / data_gt_cur[idx];
        else:
            re_err_cur[idx] = x_est_cur[idx] - data_gt_cur[idx]
    return re_err_cur
