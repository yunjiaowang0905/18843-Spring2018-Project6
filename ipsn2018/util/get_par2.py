import numpy as np
from scipy.linalg import expm, solve
from scipy.sparse import bsr_matrix
from scipy.io import savemat
from scipy.io import loadmat
from getA_c import getA_c
import matlab.engine
from cvx_solve_u import cvx_solve_u

def get_par2(A, B, x_cell):
    m, n = x_cell[0].shape
    # eng = matlab.engine.start_matlab()
    # u = eng.cvx_solve_u(A.tolist(), B.tolist(), x_cell)
    u = cvx_solve_u(A, B, x_cell)
    u_mat = np.transpose(np.reshape(u, (n,m)))
    return u_mat
