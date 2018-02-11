import numpy as np

def get_next_state(A, B, x_p, u, coe_matrix):
    m = x_p.shape[0]
    n = x_p.shape[1]
    u_vec = np.reshape(u.T, (m*n, 1))
    x_p_vec = np.reshape(x_p.T, (m*n, 1))
    x_n_vec = coe_matrix.dot(x_p_vec) + B.dot(u_vec)
    x_n = np.reshape(x_n_vec, (n, m)).T
    return x_n