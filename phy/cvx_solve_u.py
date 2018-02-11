import cvxpy
import numpy as np
from scipy.sparse import bsr_matrix
from scipy.linalg import block_diag

def cvx_solve_u(A, B, x_cell):
    T, ni, nj = x_cell.shape
    n = ni * nj
    Xvec = np.empty((0,n))

    for id in range(T):
        xt = np.transpose(x_cell[T, :, :])
        Xvec = np.append(Xvec, xt.flatten(), axis=0)

    print("shape of Xvec: ", Xvec.shape)

    BigA = A = bsr_matrix(A)
    BigB = B = bsr_matrix(B)

    for id in range(T-2):
        BigA = block_diag(BigA, A)
        BigB = block_diag(BigB, B)

    u = cvxpy.Variable(n)
    objective = cvxpy.Minimize(cvxpy.sum_squares(Xvec[n+1:n*T] - BigA * Xvec[1:n*(T-1) - BigB * np.tile(u, (T-1,1))]))
    result = cvxpy.Problem(objective).solve()

    return u.value
