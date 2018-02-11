import numpy as np
import math as math
from .get_next_state import get_next_state
from .get_par2 import get_par2
import config

def particle_filter_run(z, pf_upd_flag_cur, i_t, x, pw, xp, umat):
    u_mat_tmp = get_par2(config.K, config.vx, config.vy, config.l, config.dt, x[i_t-config.t_len_his:i_t-1])
    umat[i_t] = u_mat_tmp
    x_P_update = np.zeros((config.N_pf, config.n_lat, config.n_lon))
    for i_pf in range (0, config.N_pf):
        x_est_pre = xp[i_t-1][i_pf]
        x_P_update[i_pf] = get_next_state(config.A, config.B, x_est_pre, u_mat_tmp,config.coe_matrix) + math.sqrt(config.x_N_pf) * np.random.randn(config.n_lat, config.n_lon)
    return [x,pw,xp,umat,pf_upd_flag_cur]