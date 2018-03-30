import numpy as np
from cvx_solve_u import cvx_solve_u

def get_par2(A, B, x_cell):
    m, n = x_cell[0].shape
    u = cvx_solve_u(A, B, x_cell)
    u_mat = np.transpose(np.reshape(u, (n,m)))
    return u_mat