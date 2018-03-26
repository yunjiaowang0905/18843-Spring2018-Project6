import numpy as np
import math as math
from get_next_state import get_next_state
from get_par2 import get_par2
from draw import draw_heatmap
import config

def particle_filter_run(z, pf_upd_flag_cur, i_t, x, pw, xp, umat):
    if i_t < config.t_len_his:
        xp[i_t] = np.tile(config.get_X0(), (100, 1, 1)) + math.sqrt(config.V_pf) * np.random.randn(config.N_pf, config.n_lat, config.n_lon)
        x[i_t] = config.get_X0()
        pw[i_t] = np.ones((100, config.n_lat, config.n_lon))/100
        #pf_upd_flag_cur initialize with 0
        print("Initialization")
    else:
        print("--------------------",i_t)
        u_mat_tmp = get_par2(config.K, config.vx, config.vy, config.l, config.dt, x[i_t-config.t_len_his:i_t])
        umat[i_t] = u_mat_tmp
        # draw_heatmap(u_mat_tmp, i_t)
        x_P_update = np.zeros((config.N_pf, config.n_lat, config.n_lon))
        for i_pf in range (0, config.N_pf):
            x_est_pre = xp[i_t-1][i_pf]
            x_P_update[i_pf] = get_next_state(config.A, config.B, x_est_pre, u_mat_tmp,config.coe_matrix) + math.sqrt(config.x_N_pf) * np.random.randn(config.n_lat, config.n_lon)
        np.nan_to_num(x_P_update)
        x_P_update[x_P_update < 0] = 0
        x_P_update[x_P_update > config.gas_max_val] = config.gas_max_val
        sample_wrong = 0
        sample_count = 0
        for i in range(0, config.N_pf):
            for j in range(0, config.n_lat):
                for k in range(0, config.n_lon):
                    if (z[j,k] > 0):
                        sample_wrong += abs(x_P_update[i,j,k]-z[j,k])/z[j,k]
                        sample_count += 1
                        x_P_update[i,j,k] = z[j,k]
        if (sample_count != 0):
            print("sample_worng_rate =", sample_wrong/sample_count)
        else:
            print("sample_worng_rate =", 0)
        #print("sample_count =", sample_count)
        xp[i_t] = x_P_update
        test = x_P_update.mean(0)
        x[i_t] = test
    #print("----------x----------", i_t)
    draw_heatmap(x[i_t], i_t)
    #print(x[i_t])
    return [x,pw,xp,umat,pf_upd_flag_cur]