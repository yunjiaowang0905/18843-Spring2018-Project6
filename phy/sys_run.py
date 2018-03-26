import numpy as np
from particle_filter_run import particle_filter_run
import config

def sys_run(i_t):
    # prepare data
    tmp = config.get_x_est_gp()[i_t]
    tmp[tmp < 0] = 0
    tmp[tmp >= config.pre_max_val] = config.pre_max_val
    config.set_data_upd_interp(i_t, tmp)
    # x_est_adp empty
    if i_t < config.t_len_his + 1:
        config.set_X0(tmp)
        # config.set_X0(config.get_data_upd_interp()[i_t])
    result = particle_filter_run(tmp,config.get_pf_upd_flag_adp(),i_t,config.get_x_est_adp(),config.get_P_w_adp(),config.get_x_P_adp(),config.get_u_mat_adp())
    config.set_x_est_adp(result[0])
    config.set_P_w_adp(result[1])
    config.set_x_P_adp(result[2])
    config.set_u_mat_adp(result[3])
    config.set_pf_upd_flag_adp(result[4])