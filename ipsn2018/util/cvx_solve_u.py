__author__ = 'Nanshu Wang'
import cvxpy
import numpy as np
from scipy.sparse import bsr_matrix
from scipy.sparse import block_diag

def cvx_solve_u(A, B, x_cell):
    T = len(x_cell)
    ni, nj = x_cell[0].shape
    n = ni * nj
    Xvec = np.empty((0,n))

    for id in range(T):
        xt = np.transpose(x_cell[id])
        Xvec = np.append(Xvec, [xt.flatten()], axis=0)

    BigA = A = bsr_matrix(A)
    BigB = B = bsr_matrix(B)

    for id in range(T-2):
        BigA = block_diag((BigA, A))
        BigB = block_diag((BigB, B))

    u = cvxpy.Variable(n)

    objective = cvxpy.Minimize(cvxpy.norm( Xvec.flatten()[n:n*T] - BigA * (Xvec.flatten()[0:n*(T-1)]) - BigB * cvxpy.vstack(u,u) ))

    # TODO: Change solver and tolerance levels
    result = cvxpy.Problem(objective).solve(solver=cvxpy.ECOS, verbose=False, abstol=1e-3, reltol=1e-3, feastol=1e-3)

    return u.value