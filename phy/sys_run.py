import numpy as np
from .particle_filter_run import particle_filter_run
import config

def sys_run(i_t):
    # prepare data
    tmp = config.get_x_est_gp()[i_t]
    tmp[tmp < 0] = 0
    tmp[tmp >= pre_max_val] = pre_max_val
    config.set_data_upd_interp(i_t, tmp)
    # x_est_adp empty
    particle_filter_run(config.get_data_upd_interp()[i_t],config.get_pf_upd_flag_adp(),i_t,config.get_x_est_adp(),config.get_P_w_adp(),config.get_x_P_adp(),config.get_u_mat_adp())