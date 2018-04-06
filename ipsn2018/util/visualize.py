from __future__ import absolute_import, division, print_function
__author__ = 'Nanshu Wang'
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def show_maps(preds, row, col, title, i_t, name):
    fig, axes = plt.subplots(row, col)

    for r in range(row):
        for c in range(col):
            i = r*row+c
            cax = axes[r,c].matshow(np.transpose(preds[i]), vmin=0, vmax=2.5, cmap = cm.get_cmap("terrain"))
            plt.sca(axes[r,c])
            # plt.xticks([0,8,16,24,32,40,48,56,64])
            # plt.yticks([0,8,16])
            plt.xticks(np.arange(1,10,60))
            plt.yticks(np.arange(1,10,12))
            plt.title(title[i], y=2)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cax, cax=cbar_ax)

    # fig.show()
    # fig.savefig("data/fig/" + name + str(i_t) + ".png")
    plt.close()

def show_map(data, title):
    plt.matshow(data, vmin=0, vmax=2.5, cmap = cm.get_cmap("terrain"))
    plt.title(title, y=1.08)
    plt.colorbar()
    # plt.savefig("data/fig/" + title + ".png")
    plt.close()

def show_result(data, n_time):
    i_t = 0
    # while i_t < n_time:
    #     if i_t >= 1:
    #         # gt_l = data.data_gt[i_t-1]
    #         gt_r = data.data_gt[i_t]
    #         # pred_l = data.x_est_adp[i_t-1]
    #         pred_r = data.x_est_adp[i_t]
    #         ann_r = data.x_est_ann[i_t]
    #         gp_r = data.x_est_gp[i_t]
    #         show_maps([gt_r, pred_r, ann_r, gp_r], 2, 2,
    #                         ["gt ", "adp", "ann", "gp"],
    #                         i_t, "gt_adp_ann_gp_")
    #
    #     i_t += 1
    # print(n_time)
    truth = data.data_gt
    smp_count = data.smp_cnt_gt
    n_time = 515
    calc_abs_error("adp", truth[3:], data.x_est_adp[3:], smp_count[3:], n_time)
    calc_abs_error("ann", truth[3:], data.x_est_ann[3:], smp_count[3:], n_time)
    calc_abs_error("gp", truth[3:], data.x_est_gp[3:], smp_count[3:], n_time)
    calc_rel_error("adp", truth[3:], data.x_est_adp[3:], smp_count[3:], n_time)
    calc_rel_error("ann", truth[3:], data.x_est_ann[3:], smp_count[3:], n_time)
    calc_rel_error("gp", truth[3:], data.x_est_gp[3:], smp_count[3:], n_time)

def calc_abs_error(alg, truth, pred, smp, n_time):
    i_t = 0
    total_err = []
    while i_t < n_time-3:
        error, val = get_abs_error(truth[i_t], pred[i_t], smp[i_t])
        # show_map(error, alg + " abs error" + str(i_t))
        #print("average relative error for time %d: %f\n" % (i_t, val))
        total_err += val
        i_t += 1
    print(alg + " average abs error % f\n" % np.mean(total_err))

def calc_rel_error(alg, truth, pred, smp, n_time):
    i_t = 0
    total_err = []
    while i_t < n_time-3:
        error, val = get_rel_error(truth[i_t], pred[i_t], smp[i_t])
        # show_map(error, alg + " abs error" + str(i_t))
        #print("average relative error for time %d: %f\n" % (i_t, val))
        total_err += val
        i_t += 1
    print(alg + " average rel error % f\n" % np.mean(total_err))

def get_rel_error(truth, pred, smp_count):
    row, col = pred.shape
    total = []
    error = np.zeros((row, col))
    for r in range(row):
        for c in range(col):
            if smp_count[r][c] > 0:
                error[r][c] = abs(pred[r][c] - truth[r][c]) / truth[r][c]
                total.append(error[r][c])
    return error, total

def get_abs_error(truth, pred, smp_count):
    row, col = pred.shape
    total = []
    error = np.zeros((row, col))
    for r in range(row):
        for c in range(col):
            if smp_count[r][c] > 0:
                error[r][c] = abs(pred[r][c] - truth[r][c])
                total.append(error[r][c])
    return error, total