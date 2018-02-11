import numpy as np

def loc(i, j, n):
    return n * (i - 1) + j

def getA_c(K, l, Vx, Vy):
    m, n = Vx.shape
    A = np.zeros((m,n))

    for i in range(m):
        for j in range(n):
            tmp = -4 * K / (l*l)

            if j > 0 and j < n and i > 0 and i < m:
                tmp -= ( (Vx[i,j+1] - Vx[i,j-1]) + (Vy[i,j+1] - Vy[i,j-1]) ) / (2*l)

            if j > 0:
                A[loc(i,j,n), loc(i,j-1,n)] = K / (l*l) + Vx[i,j] / (2*l)

            if j < n:
                A[loc(i,j,n), loc(i,j+1,n)] = K / (l*l) + Vx[i,j] / (2*l)

            if i > 0:
                A[loc(i,j,n), loc(i-1,j,n)] = K / (l*l) + Vy[i,j] / (2*l)

            if i < m:
                A[loc(i,j,n), loc(i+1,j,n)] = K / (l*l) + Vy[i,j] / (2*l)

    return A