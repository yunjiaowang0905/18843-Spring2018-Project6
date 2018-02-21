__author__ = 'Jin Gong'
import numpy as np
from scipy.spatial.distance import cdist

def data_seperation(data_in, smp_cnt, alg_sep, i_t, pct_gt, n_lat, n_lon, pct_mat, flag_empty):
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
    row, col = np.nonzero(smp_cnt)
    n_val = len(row)

    # for groung truth
    data_gt = np.zeros(n_lat, n_lon)
    smp_cnt_gt = np.zeros(n_lat, n_lon)

    # for prediction model
    data_pre = np.zeros(n_lat, n_lon)
    smp_cnt_pre = np.zeros(n_lat, n_lon)

    # for update
    data_upd = np.zeros(n_lat, n_lon)
    smp_cnt_upd = np.zeros(n_lat, n_lon)

    # for online verification
    data_ver = np.zeros(n_lat, n_lon)
    smp_cnt_ver = np.zeros(n_lat, n_lon)

    n_gt = round(pct_gt * n_val)
    r_all = np.random.permutation(n_val)
    n_pre = round((n_val - n_gt) * pct_mat[1])
    n_upd = round((n_val - n_gt) * pct_mat[2])
    n_ver = n_val - n_gt - n_pre - n_upd

    if not flag_empty:
        # choose different area, each area get average value
        if alg_sep == 1:
            r_gt = np.sort(r_all[0 : n_gt])
            r_pre = sort(r_all[n_gt : n_gt + n_pre + 1])
            r_upd = sort(r_all[n_gt+n_pre: n_gt + n_pre + n_upd + 1])
            r_ver = sort(r_all[n_gt + n_pre + n_upd: end + 1]

            for i_gt in xrange(len(r_gt)):
                id_cur = r_gt[i_gt]
                r = row[id_cur]
                c = col[id_cur]
                data_gt[r][c] = np.mean(data_in[r][c])
                smp_cnt_gt[r][c] = smp_cnt[r][c]

            for i_pre in xrange(n_pre):
                id_cur = r_pre[i_pre]
                r = row[id_cur]
                c = col[id_cur]
                data_pre[r][c] = np.mean(data_in[row[r][c])
                smp_cnt_pre[r][c] = smp_cnt[r][c]

            for i_upd in xrange(n_upd):
                id_cur = r_upd[i_upd]
                r = row[id_cur]
                c = col[id_cur]
                data_upd[r][c] = np.mean(data_in[[r][c])
                smp_cnt_upd[r][c] = smp_cnt[r][c]

            for i_ver in xrange(n_ver):
                id_cur = r_ver[i_ver]
                r = row[id_cur]
                c = col[id_cur]
                data_ver[r][c] = np.mean(data_in[r][c])
                smp_cnt_ver[r][c] = smp_cnt[r][c]

        else if alg_sep == 2:
            c_point = [n_lat/2 - 1, n_lon/2 - 1] # python is 0-based, for caculate the distance, minus 1 first
            dis_mat = cdist(np.transpose(row,col), [c_point])
            if i_t % 2 == 1:
                indices = np.argsort(dis_mat)[::-1] # descending order
            else:
                indices = np.argsort(dis_mat)

            for i_gt in xrange(n_gt):
                r = row[indices[i_gt]]
                c = col[indices[i_gt]]
                data_gt[r][c] = np.mean(data_in[r][c])  # not sure about the origin matlab code
                smp_cnt_gt[r][c] = smp_cnt[r][c]

            r_tr = np.random.permutation(n_val - n_gt) + n_gt # not sure about the origin matlab code

            for i_pre in xrange(n_pre):
                id_cur = r_tr[i_pre]
                r = row[indices[id_cur]]
                c = col[indices[id_cur]]
                data_pre[r][c] = np.mean(data_in[r][c])
                smp_cnt_pre[r][c] = smp_cnt[r][c]

            for i_upd in xrange(n_pre):
                id_cur = r_tr[i_upd + n_pre]
                r = row[indices[id_cur]]
                c = col[indices[id_cur]]
                data_upd[r][c] = np.mean(data_in[r][c])
                smp_cnt_upd[r][c] = smp_cnt[r][c]

            for i_ver in xrange(n_pre):
                id_cur = r_tr[i_ver + n_pre + n_upd]
                r = row[indices[id_cur]]
                c = col[indices[id_cur]]
                data_ver[r][c] = np.mean(data_in[r][c])
                smp_cnt_ver[r][c] = smp_cnt[r][c]

    return data_gt, data_pre, data_upd, data_ver, smp_cnt_gt, smp_cnt_pre, smp_cnt_upd, smp_cnt_ver
