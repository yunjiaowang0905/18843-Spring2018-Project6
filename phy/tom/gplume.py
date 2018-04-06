import numpy as np
from scipy.io import loadmat
from scipy.linalg import expm, solve, block_diag, norm
from scipy.optimize import minimize
from scipy.stats import linregress
import matplotlib.pyplot as plt
import math, cvxpy

def gplume(x, y, Q, U=5):
    ay, by = 0.34, 0.82
    sigmay = ay * np.power(np.absolute(x), by) * (x > 0)
    C = Q / (math.pi * U * sigmay) * np.exp(-0.5 * np.square(y) / np.square(sigmay))
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    return C

def getDistanceVector(x, y, nx, ny, l):
    x_dist = np.zeros((nx, ny))
    y_dist = np.zeros((nx, ny))

    for i in xrange(nx):
        for j in xrange(ny):
            x_dist[i,j] = (i-x) * l
            y_dist[i,j] = (j-y) * l

    return x_dist.reshape((1, nx*ny)), y_dist.reshape((1, nx*ny))

def get_P(m, n, l=500):
    P = np.zeros((m*n, m*n))
    Q0 = 1

    for i in xrange(m):
        for j in xrange(n):
            x_dist, y_dist = getDistanceVector(i, j, m, n, l)
            P[i*n+j, :] = gplume(x_dist, y_dist, Q0) / Q0

    return P

def predictSource(C_last, C_curr, l=500):
    n_lat, n_lon = C_last.shape
    C_interp_vec = C_last.reshape((1, n_lat*n_lon))

    P = np.zeros((n_lat*n_lon, n_lat*n_lon))

    Q0 = 1

    # TODO: test for now
    for i in xrange(n_lat):
        for j in xrange(n_lon):
            x_dist, y_dist = getDistanceVector(i, j, n_lat, n_lon, l)
            P[i*n_lon+j, :] = gplume(x_dist, y_dist, Q0) / Q0

    # P = np.transpose(P)
    # Q = np.linalg.lstsq(P, C_curr.reshape((n_lat*n_lon, 1)))[0].reshape(C_last.shape)
    # print Q
    # plt.imshow(Q, interpolation='nearest')
    # plt.show()
    
    # C_gp = np.sum(P, axis=1).reshape(C_last.shape)
    # # print C_gp
    # plt.imshow(C_gp, interpolation='nearest')
    # plt.show()


                   
