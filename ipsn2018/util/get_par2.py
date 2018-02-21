__author__ = 'Nanshu Wang'
import numpy as np
from scipy.linalg import expm, solve
from scipy.sparse import bsr_matrix
from getA_c import getA_c
from cvx_solve_u import cvx_solve_u

def get_par2(K, vx, vy, l, dt, x_cell):
    m, n = x_cell[0].shape
    V_x = np.zeros((m,n)) + vx
    V_y = np.zeros((m,n)) + vy

    A = getA_c(K, l, V_x, V_y)
    A -= np.diag(np.sum(A, axis=1))
    A -= np.diag(np.diag(A)) * 1e-6

    A1 = expm(A * dt)
    B = solve(A, A1 - np.eye(A1.shape[0]))
    A1[np.where(np.absolute(A1) < 1e-3 * np.amax(np.absolute(A1)))] = 0
    B[np.where(np.absolute(B) < 1e-3 * np.amax(np.absolute(B)))] = 0

    coe_matrix = bsr_matrix(A1)
    B = bsr_matrix(B)

    u = cvx_solve_u(coe_matrix, B, x_cell)
    u_mat = np.transpose(np.reshape(u, (n,m)))
    return u_mat
