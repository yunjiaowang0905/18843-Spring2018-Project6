import numpy as np
from scipy.linalg import expm, solve
from scipy.sparse import bsr_matrix
from getA_c import getA_c

# Result path
RESULT_PATH = "../result/gt_40_upd100_sep2sw_tlenhis3_sr16_tr1_cn29_early/gt_40_upd100_sep2_tlenhis3_rangeLim0_ann_dd_gp.mat"
NEWDATA_PATH = "../result/distribution_res_newArea.csv"

n_time = 5903
n_lat = 12
n_lon = 60
N_pf = 100
pre_max_val = 10
l = 500 # ???
res_t = 1
dt = res_t * 60 *60
t_len_his = 100
alg_upd = 4
flag_range_lim = 1
flag_cal_feature = 1
K = 100
vx = 0
vy = 0
x_N_pf = 0.5710**2
run_alg = 1
V_pf = 0.3260
gas_max_val = 60

def init():
    global coe_matrix
    coe_matrix = np.full((720, 720), 0)
    global x_est_gp
    x_est_gp = np.zeros((n_time,n_lat,n_lon), dtype=np.float64)
    global data_upd_interp
    data_upd_interp = np.zeros((n_time,n_lat,n_lon), dtype=np.float64)
    global pf_upd_flag_adp
    pf_upd_flag_adp = np.zeros((n_time,n_lat,n_lon), dtype=np.float64) # unneccssary
    global x_est_adp
    x_est_adp = np.zeros((n_time,n_lat,n_lon), dtype=np.float64)
    global P_w_adp
    P_w_adp = np.zeros((n_time,100,n_lat,n_lon), dtype=np.float64) # unneccssary
    global x_P_adp
    x_P_adp = np.zeros((n_time,100,n_lat,n_lon), dtype=np.float64) # unneccssary
    global u_mat_adp
    u_mat_adp = np.zeros((n_time,n_lat,n_lon), dtype=np.float64) # unneccssary
    global X0
    X0 = np.zeros((n_lat,n_lon), dtype=np.float64)

    V_x = np.zeros((n_lat,n_lon)) + vx
    V_y = np.zeros((n_lat,n_lon)) + vy

    #initialize A B, similar to getPar2
    global A
    A = getA_c(K, l, V_x, V_y)
    A -= np.diag(np.sum(A, axis=1))
    A -= np.diag(np.diag(A)) * 1e-6

    A1 = expm(A * dt)
    global B
    B = solve(A, A1 - np.eye(A1.shape[0]))
    A1[np.where(np.absolute(A1) < 1e-3 * np.amax(np.absolute(A1)))] = 0
    B[np.where(np.absolute(B) < 1e-3 * np.amax(np.absolute(B)))] = 0

    coe_matrix = bsr_matrix(A1)
    B = bsr_matrix(B)

def get_A():
    return A

def get_B():
    return B

def get_coe_matrix():
    return coe_matrix

def get_pf_upd_flag_adp():
    return pf_upd_flag_adp

def get_x_est_adp():
    return x_est_adp

def get_P_w_adp():
    return P_w_adp

def get_x_P_adp():
    return x_P_adp

def get_u_mat_adp():
    return u_mat_adp

def get_x_est_gp():
    return x_est_gp

def get_data_upd_interp():
    return data_upd_interp

def get_X0():
    return X0

def set_X0(value):
    global X0
    X0 = value

def set_x_est_gp(i, value):
    global x_est_gp
    x_est_gp[i] = value

def set_x_est_gp_single(i, x, y, value):
    global x_est_gp
    x_est_gp[i][x][y] = value

def set_data_upd_interp(i, value):
    global data_upd_interp
    data_upd_interp[i] = value

def set_x_est_adp(value):
    global x_est_adp
    x_est_adp = value

def set_P_w_adp(value):
    global P_w_adp
    P_w_adp = value

def set_x_P_adp(value):
    global x_P_adp
    x_P_adp = value

def set_u_mat_adp(value):
    global u_mat_adp
    u_mat_adp = value

def set_pf_upd_flag_adp(value):
    global pf_upd_flag_adp
    pf_upd_flag_adp = value




