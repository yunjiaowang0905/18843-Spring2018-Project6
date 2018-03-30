__author__ = 'Nanshu Wang'
import cvxpy
import numpy as np
from scipy.sparse import block_diag
import copy

def cvx_solve_u(A, B, x_cell):
    T = len(x_cell)
    ni, nj = x_cell[0].shape
    n = ni * nj
    Xvec = np.empty((0,n))

    for id in range(T):
        xt = np.transpose(x_cell[id])
        Xvec = np.append(Xvec, [xt.flatten()], axis=0)


    BigA = copy.deepcopy(A)
    BigB = copy.deepcopy(B)

    for id in range(T-2):
        BigA = block_diag((BigA, A))
        BigB = block_diag((BigB, B))

    u = cvxpy.Variable(n)
    uu = u
    for i in range(T-2):
        uu = cvxpy.vstack(uu,u)
    Xvec =  Xvec.flatten()
    coef = Xvec[n:n*T] - BigA * Xvec[0:n*(T-1)]
    fn = cvxpy.norm( coef - BigB.toarray() * uu )
    objective = cvxpy.Minimize(fn)
    eta = 2.22*1e-16
    soltol = eta**(3/8)
    stdtol = eta**(1/4)
    redtol = eta**(1/4)
    prob = cvxpy.Problem(objective)
    # TODO: Change solver and tolerance levels
    prob.solve(solver=cvxpy.ECOS, verbose=False, abstol=1e-3, reltol=1e-3, feastol=1e-3)
    # print prob.value
    return u.value