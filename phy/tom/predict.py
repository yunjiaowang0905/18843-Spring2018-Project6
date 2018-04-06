import numpy as np
from scipy.io import loadmat, savemat
from scipy.linalg import expm, solve, block_diag, norm
from scipy.optimize import minimize
from scipy.stats import linregress
import matplotlib.pyplot as plt
import math, time

def loc(i, j, n):
    return n * i + j

def removeInvalidValues(C):
    C[np.where(C < 0)] = 0
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0

def visualizeConcentration(orig, interp, pred, it, save_fig=True):
    plt.subplot(131)
    plt.imshow(orig, vmin=0, vmax=50, interpolation='nearest')
    plt.subplot(132)
    plt.imshow(interp, vmin=0, vmax=50, interpolation='nearest')
    plt.subplot(133)
    plt.imshow(pred, vmin=0, vmax=50, interpolation='nearest')

    if save_fig:
        plt.savefig('../matdata/predict/data_predict_{}.jpg'.format(it))
        plt.clf()
    else:
        plt.show()
    


# Compute Coefficient Matrices A and B
def getCoefficients(m, n, K=1.0, l=500, dt=3600):
    A0 = np.zeros((m*n,m*n))

    for i in range(m):
        for j in range(n):
            A0[loc(i, j, n), loc(i, j, n)] = -4 * K / (l*l)

            if j > 0:
                A0[loc(i,j,n), loc(i,j-1,n)] = K / (l*l)

            if j < n-1:
                A0[loc(i,j,n), loc(i,j+1,n)] = K / (l*l)

            if i > 0:
                A0[loc(i,j,n), loc(i-1,j,n)] = K / (l*l)

            if i < m-1:
                A0[loc(i,j,n), loc(i+1,j,n)] = K / (l*l)
    # A0 = get_P(m, n)
    A = expm(A0 * dt)
    B = solve(A0, A - np.eye(A0.shape[0]))

    return A, B

def getFilterMatrix(data):
    n_lat, n_lon = data.shape
    data_vec = data.reshape((n_lat*n_lon,))
    F = np.eye(n_lat*n_lon)
    points = np.where(data_vec < 0)
    F[points, points] = 0
    return F

def findLinearRegression(orig, pred):
    points = np.where(orig >= 0)
    x, y = pred[points], orig[points]

    if len(x) == 0:
        return pred
    
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    pred = slope * pred + intercept
    removeInvalidValues(pred)
    return pred

def predictSource(C_last, C_curr, C_next, C_train):
    n_lat, n_lon = C_curr.shape
    A, B = getCoefficients(n_lat, n_lon)

    C_last_vec = C_last.reshape((n_lat*n_lon,))
    C_curr_vec = C_curr.reshape((n_lat*n_lon,))
    C_next_vec = C_next.reshape((n_lat*n_lon,))

    C_train_vec = C_train.reshape((n_lat*n_lon,))
    C_train_vec[np.where(C_train_vec < 0)] = 0

    v1 = np.append(C_curr_vec, C_last_vec)
    v2 = np.append(C_next_vec, C_curr_vec)

    rhs1 = v1 - block_diag(A, A).dot(v2)

    F = getFilterMatrix(C_train)
    FA = F.dot(A)
    FB = F.dot(B)
    rhs2 = FA.dot(C_curr_vec) - C_train_vec

    bigB = np.append(B, B, axis=0)
    bigB = np.append(bigB, FB, axis=0)
    rhs = np.append(rhs1, rhs2, axis=0)
    
    U = np.linalg.lstsq(bigB, rhs)
    return U[0], A, B

    # U = cvxpy.Variable(n_lat*n_lon)
    # objective = cvxpy.Minimize( cvxpy.norm(rhs - bigB*U) + cvxpy.norm(FA.dot(C_curr_vec) + FB.dot(U) - C_train_vec) )
    # result = cvxpy.Problem(objective).solve()
    # U0 = np.ones((n_lat*n_lon,))

    # objective = lambda U: norm(rhs - bigB.dot(U))
    # constraint = {
    #     'type': 'eq',
    #     'fun': lambda U: FA.dot(C_curr_vec) + FB.dot(U) - C_train_vec
    # }
    # res = minimize(objective, U0, constraints=constraint)
    # U = res.x
    # y1 = FA.dot(C_curr_vec) + FB.dot(U) - C_train_vec
    # return res.x

def predictTimeSlice(C_last, C_curr, C_next, C_train):
    n_lat, n_lon = C_curr.shape

    C_curr_vec = C_curr.reshape((n_lat*n_lon,))
    C_next_vec = C_next.reshape((n_lat*n_lon,))

    U, A, B = predictSource(C_last, C_curr, C_next, C_train)
    C = A.dot(C_next_vec) + B.dot(U)
    removeInvalidValues(C)

    C_pred = C.reshape(C_curr.shape)
    # C_pred = findLinearRegression(C_train, C_pred)
    return C_pred

def predictSequence(data_train, data_interp, time_range=None, visualize=False, save_fig=True):
    nt, n_lat, n_lon = data_train.shape

    if time_range == None:
        start_t, max_t = 0, nt
    else:
        start_t, max_t = time_range
    
    data_pred = np.zeros((max_t, n_lat, n_lon))

    data_pred[start_t, :, :] = C_last = data_interp[start_t, :, :]
    data_pred[start_t+1, :, :] = C_curr = data_interp[start_t+1, :, :]

    start = time.time()

    for it in range(start_t+2, max_t):
        C_next = data_interp[it, :, :]
        data_original = data_train[it, :, :]

        C_pred = predictTimeSlice(C_last, C_curr, C_next, data_original)
        data_pred[it, :, :] = C_pred

        data_original[np.where(data_original<0)] = float('nan')

        if visualize:
            visualizeConcentration(data_original, C_next, C_pred, it, save_fig)

        if it % 100 == 0:
            end = time.time()
            remaining = round((end-start) * (max_t-it) / (it-start_t) / 60, 1)
            print '{} of {} Time Slices Completed, Estimated Remaining Time: {} min'.format(it-start_t, max_t-start_t, remaining)

        C_last = C_curr
        C_curr = C_pred

    return data_pred

if __name__ == "__main__":
    # TODO: change these file paths if necessary

    # Training data
    data_train = loadmat('../matdata/data_train.mat')['data']

    # Interpolated Training Data
    data_interp = loadmat('../matdata/data_interp_train.mat')['data']

    # Test data
    data_test = loadmat('../matdata/data_test.mat')['data']

    # Predict concentration, and save to file
    data_pred = predictSequence(data_train, data_interp)
    savemat('../matdata/data_pred.mat', {'data': data_pred})

    # Report Error
    from computeError import computeError
    relErr, rmse = computeError(data_test, data_pred)
    print "Relative Error: ", relErr
    print "RMSE: ", rmse
