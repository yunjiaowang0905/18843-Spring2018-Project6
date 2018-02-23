import numpy as np
from scipy.linalg import expm, solve
from scipy.sparse import bsr_matrix
from scipy.io import savemat
from scipy.io import loadmat
from getA_c import getA_c
import matlab.engine
#from cvx_solve_u import cvx_solve_u

def get_par2(K, vx, vy, l, dt, x_cell):
    m, n = x_cell[0].shape
    print(len(x_cell.tolist()))
    print(len(x_cell[0].tolist()))
    print(len(x_cell[0][0].tolist()))
    V_x = np.zeros((m,n)) + vx
    V_y = np.zeros((m,n)) + vy

    A = getA_c(K, l, V_x, V_y)
    A -= np.diag(np.sum(A, axis=1))
    A -= np.diag(np.diag(A)) * 1e-6

    A1 = expm(A * dt)
    B = solve(A, A1 - np.eye(A1.shape[0]))
    A1[np.where(np.absolute(A1) < 1e-3 * np.amax(np.absolute(A1)))] = 0
    B[np.where(np.absolute(B) < 1e-3 * np.amax(np.absolute(B)))] = 0

    # coe_matrix = bsr_matrix(A1, (1024,1024))
    # B = bsr_matrix(B, (1024,1024))

    eng = matlab.engine.start_matlab()
    u = eng.cvx_solve_u(A1.tolist(), B.tolist(), x_cell.tolist())
    # u = cvx_solve_u(coe_matrix, B, x_cell)
    u_mat = np.transpose(np.reshape(u, (n,m)))
    return u_mat

# For testing get_par2()
if __name__ == "__main__":
    K = 100
    vx = vy = 0
    l = 500
    dt = 3600

    x_cell = loadmat('../x_cell.mat')['x_cell']

    cells = []
    for c in x_cell:
        cells.append(c[0])

    u_mat = get_par2(K, vx, vy, l, dt, cells)
    print(u_mat)