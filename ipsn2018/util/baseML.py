__author__ = 'Jin Gong'
import numpy as np

def baseML(data_cur, i_t, n_lat, n_lon, alg_sep, pct_mat, flag_empty, pre_max_val, data, conf):
    """
    system Running in time series
    data_cur: x * 7, numpy array
    parameters: data_rcd: data from records
                alg_grid: method to combine data in each grid
                n_lat, n_lon: latitude and longitude grids number
                i_gas: gas index, co 3, co2 4, o3 5, pm25 6
    """

    # put record data into each grid
    data_grid_cur，data.smp_cnt[i_t] = rcd2grid(data_cur, n_lat, n_lon, flag_empty)

    # seperate data into 4 parts
    data.data_gt[i_t], data.data_pre[i_t], data.data_upd[i_t], data.data_ver[i_t], \
    data.smp_cnt_gt[i_t], data.smp_cnt_pre[i_t], data.smp_cnt_upd[i_t], data.smp_cnt_ver[i_t] \
    = data_seperation(data_grid_cur, data.smp_cnt[i_t], alg_sep, i_t, pct_gt, n_lat, n_lon, pct_mat, flag_empty)

    # get the round truth when the data is flag_empty
    if np.count_nonzero(data.smp_cnt_gt[i_t]) < 3:
        data.data_gt[i_t] = data_gt[i_t - 1]
        data.smp_cnt_gt[i_t] = data.smp_cnt_gt[i_t - 1]

    # get result for data driven method
    data_tr = [sum(x) for x in zip(data.data_pre[i_t], data.data_udp[i_t], data.data_ver[i_t])] # for numpy array: data.data_pre[i_t] + data.data_udp[i_t] + data.data_ver[i_t]
    data.x_est_dd = get_dd_result(data_tr)
    # TODO:implement get_dd_redult

    if not flag_empty:
        if np.count_nonzero(data.x_est_dd[i_t]) / (n_lat * n_lon) <= 0.2:
            idx = data.x_est_dd[i_t] == 0
        if i_t > 0:
            # to modify
            r, c = np.shape(data.x_est_dd[i_t])
            for i in xrange(r):
                for j in xrange(c):
                    if data.x_est_dd[i_t][r][c] == 0:
                        data.x_est_dd[i_t][r][c] = data.x_est_dd[i_t - 1][r][c]
    else:
        data.x_est_dd[i_t] = data.x_est_dd[i_t - 1]

    if i_t < conf.t_len_his:
        data.x_est_ann[i_t] = data.x_est_dd[i_t]
        data.x_est_gp[i_t] = data.x_est_dd[i_t]
    else:
        sum_dd = np.zeros(n_lat, n_lon)
        # Creating a fitting network
        hiddenLayerSize = 10
        # TODO: implement neural network by keras


        inputs = []
        targets = []
        for t_cur in xrange (i_t - conf.t_len_his - 1, i_t - 1):
            sum_dd = [sum(x) for x in zip(sum_dd, data.x_est_dd[t_cur])]
            smp_cnt_gt_cur = data.smp_cnt_gt[t_cur]
            idx = find(smp_cnt_gt_cur > 0)
            # TODO:implement find function
            if idx:
                row,col = np.nonzero(idx)
                # TODO: modify data structure for neural network
                # inputs_cur = [t_cur * ones(1,length(idx));row';col']
                # inputs = [inputs,inputs_cur];
                # targets = [targets, (data_gt{t_cur}(idx))'];


        # TODO: train the neural networking

        # TODO: train the Gaussian gaussian_process

    if conf.flag_range_lim:
        filter_data(data.x_est_dd[i_t], pre_max_val)
        filter_data(data.x_est_ann[i_t], pre_max_val)
        filter_data(data.x_est_gp[i_t], pre_max_val)

def filter_data(data_in, pre_max_val):
    """
    filter the data out of range(0, pre_max_val)
    """
    r, c = np.shape(data_in)
    for i in xrange(r):
        for j in xrange(c):
            if data_in[r][c] < 0:
                data_in[r][c] = 0
            else if data_in[r][c] > pre_max_val:
                data_in[r][c] = pre_max_val
