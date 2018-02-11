import scipy.io as scipy
import numpy as np
import config

def main():
    config.init()
    if config.run_alg > 0:
        file_path = "/Users/youx/Documents/Mobile and Pervasive Computing/ipsn2018/result/gt_40_upd100_sep2sw_tlenhis3_sr16_tr1_cn29_early/gt_40_upd100_sep2_tlenhis3_rangeLim0_ann_dd_gp.mat"
        data = scipy.loadmat(file_path)
        tmp = np.array(data['x_est_gp'])
        for i in range(0, n_time):
            config.set_x_est_gp(i, tmp[i][0])
        for i_t in range(0, n_time):
            sys_run(i_t)

main()